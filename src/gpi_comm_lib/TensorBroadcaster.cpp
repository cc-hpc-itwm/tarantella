#include "TensorBroadcaster.hpp"

#include "distribution/utilities.hpp"
#include "gpi/Context.hpp"
#include "gpi/ResourceManager.hpp"
#include "gpi/SegmentBuffer.hpp"

#include <cstring>
#include <vector>

namespace tarantella
{
  TensorBroadcaster::TensorBroadcaster(GPI::Context& context,
                                       GPI::SegmentID segment_id,
                                       GPI::Group const& group,
                                       std::vector<collectives::TensorInfo> const& tensor_infos,
                                       GPI::Rank root_rank)
  : context(context),
    group(group),
    queue_handler(),
    root(root_rank),
    barrier(group)
  {
    if(!group.contains_rank(root_rank))
    {
      throw std::runtime_error("[TensorBroadcaster::constructor]:\
                                Incorrect root_rank is not part of the broadcast group");
    }

    auto const overhead_factor = 1.0;
    auto& resource_manager = context.get_resource_manager();
    auto const segment_size = distribution::get_segment_size(tensor_infos, overhead_factor);

    resource_manager.make_segment_resources(segment_id, group, segment_size);

    // Barrier is required, to ensure all ranks have finished registering
    // their segments to their communication partners
    barrier.blocking_barrier();

    for(auto const& info : tensor_infos)
    {
      auto const size_in_bytes = info.get_nelems() * getDataTypeSize(info.get_elem_type());
      buffers.emplace_back(resource_manager.get_buffer_of_size(segment_id, size_in_bytes));
    }

    auto const notifications = resource_manager.get_notification_range(segment_id,
                                                                       collectives::broadcast::getNumberOfNotifications(group.get_size()));
    bcast_op = std::make_unique<collectives::broadcast>(root, segment_size, segment_id, buffers.front().get_offset(),
                                                        notifications.first, queue_handler);
  }

  void TensorBroadcaster::exec_broadcast(std::vector<void*> const& data_ptrs)
  {
    // copy data to segments
    if (context.get_rank() == root)
    {
      for (std::size_t i = 0; i < data_ptrs.size(); ++i)
      {
        std::memcpy(buffers[i].get_ptr(), data_ptrs[i], buffers[i].get_size());
      }
    }

    // start the operation
    if (context.get_rank() == root)
    {
      bcast_op->signal();
    }
    // execute broadcast
    while(bcast_op->operator()() != 0);

    // copy results back to buffers
    if (context.get_rank() != root)
    {
      for (std::size_t i = 0; i < data_ptrs.size(); ++i)
      {
        std::memcpy(data_ptrs[i], buffers[i].get_ptr(), buffers[i].get_size());
      }
    }

    // finalize operation
    barrier.blocking_barrier();
  }
}

