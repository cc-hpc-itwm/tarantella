#include "TensorBroadcaster.hpp"

#include <GaspiCxx/collectives/non_blocking/collectives_lowlevel/BroadcastSendToAll.hpp>
#include <GaspiCxx/collectives/non_blocking/Broadcast.hpp>

#include <cstring>
#include <vector>

namespace tarantella
{
  using Broadcast = gaspi::collectives::Broadcast<char, gaspi::collectives::BroadcastAlgorithm::SEND_TO_ALL>;

  namespace
  {
    std::size_t get_total_size(std::vector<collectives::TensorInfo> const& DNN)
    {
      if(DNN.size() == 0)
      {
        throw std::logic_error("tarantella::get_total_size: Empty DNN provided");
      }

      auto add_tensor_size_in_bytes = [](auto sum, auto tensor_info)
                                      { return sum + tensor_info.get_size_bytes(); };
      auto const partition_size = std::accumulate(DNN.begin(), DNN.end(), 0UL, add_tensor_size_in_bytes);
      return partition_size;
    }
  }

  TensorBroadcaster::TensorBroadcaster(gaspi::group::Group const& group,
                                       std::vector<collectives::TensorInfo> const& tensor_infos,
                                       gaspi::group::Rank root_rank)
  : group(group),
    root(root_rank),
    tensor_infos(tensor_infos),
    bcast_buffer(get_total_size(tensor_infos)),
    bcast_op(std::make_unique<Broadcast>(group, bcast_buffer.size(), root))
  { }

  void TensorBroadcaster::exec_broadcast(std::vector<void*> const& data_ptrs)
  {
    // copy data to segments
    if (group.rank() == root)
    {
      auto current_buffer_index = 0UL;
      for (std::size_t i = 0; i < data_ptrs.size(); ++i)
      {
        current_buffer_index += tensor_infos[i].get_size_bytes();
        std::memcpy(&bcast_buffer[current_buffer_index], data_ptrs[i], tensor_infos[i].get_size_bytes());
      }
      bcast_op->start(bcast_buffer.data());
    }
    else
    {
      bcast_op->start();
    }

    bcast_op->waitForCompletion(bcast_buffer.data());

    // copy results back to buffers
    if (group.rank() != root)
    {
      auto current_buffer_index = 0UL;
      for (std::size_t i = 0; i < data_ptrs.size(); ++i)
      {
        current_buffer_index += tensor_infos[i].get_size_bytes();
        std::memcpy(data_ptrs[i], &bcast_buffer[current_buffer_index], tensor_infos[i].get_size_bytes());
      }
    }
  }
}

