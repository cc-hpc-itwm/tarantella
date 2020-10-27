#include "PipelineCommunicator.hpp"

#include "collectives/barrier/GPIBarrier.hpp"
#include "distribution/GroupBuilder.hpp"
#include "gpi/gaspiCheckReturn.hpp"

#include <GASPI.h>

#include <cstring>

namespace tarantella
{
  PipelineCommunicator::PipelineCommunicator(
    GPI::Context& context,
    std::unordered_map<ConnectionID, ConnectionInfo> const& connection_infos,
    std::size_t num_micro_batches)
  : resource_manager(context.get_resource_manager())
  {
    for(auto const& [conn_id, conn_info] : connection_infos)
    {
      auto const segment_id = conn_info.segment_id;
      auto const buffer_size = conn_info.microbatched_buffer_size_bytes;
      auto const segment_size = 2 * num_micro_batches * buffer_size;
    
      auto const segment_group = resource_manager.make_group({context.get_rank(), conn_info.other_rank});
      resource_manager.make_segment_resources(segment_id, segment_group, segment_size);

      std::vector<GPI::SegmentBuffer> send_bufs;
      std::vector<GPI::SegmentBuffer> recv_bufs;
      std::vector<GPI::NotificationManager::NotificationID> notifications;
      for(std::size_t m_id = 0; m_id < num_micro_batches; ++m_id)
      {
        send_bufs.push_back(resource_manager.get_buffer_of_size(segment_id, buffer_size));
        recv_bufs.push_back(resource_manager.get_buffer_of_size(segment_id, buffer_size));
        notifications.push_back(resource_manager.get_notification_range(segment_id, 1).first);
      }
      connections.emplace(conn_id, SendRecvResources(conn_info.other_rank,
                                                     send_bufs,
                                                     recv_bufs,
                                                     notifications));
    }

    // Barrier is required, to ensure all ranks have finished registering
    // their segments to their communication partners
    collectives::Barrier::GPIBarrierAllRanks barrier;
    barrier.blocking_barrier();
  }

  void PipelineCommunicator::non_blocking_send(void* local_send_buf,
                                               ConnectionID conn_id,
                                               MicrobatchID micro_id)
  {
    auto const& local_segment_buf = connections[conn_id].send_bufs[micro_id];
    auto const& remote_segment_buf = connections[conn_id].recv_bufs[micro_id];

    copy_data_to_segment(local_send_buf, local_segment_buf);

    GPI::gaspiCheckReturn(
      gaspi_write_notify(local_segment_buf.get_segment_id(),
                         local_segment_buf.get_offset(),
                         connections[conn_id].other_rank,
                         remote_segment_buf.get_segment_id(),
                         remote_segment_buf.get_offset(),
                         local_segment_buf.get_size(),
                         connections[conn_id].notifications[micro_id],
                         micro_id + 1, // to check micro_id at recv (must not be zero)
                         resource_manager.get_queue_id_for_write_notify(),
                         GASPI_BLOCK),
      "PipelineCommunicator::non_blocking_send");
  }

  void PipelineCommunicator::blocking_recv(void* local_recv_buf,
                                           ConnectionID conn_id,
                                           MicrobatchID micro_id)
  {
    auto const& local_segment_buf = connections[conn_id].recv_bufs[micro_id];
    gaspi_notification_id_t received_notification_id = 0;
    gaspi_notification_t received_notification_value = 0;

    GPI::gaspiCheckReturn(
      gaspi_notify_waitsome(local_segment_buf.get_segment_id(),
                            connections[conn_id].notifications[micro_id],
                            1,
                            &received_notification_id,
                            GASPI_BLOCK),
      "PipelineCommunicator::blocking_recv : gaspi_notify_waitsome");
    GPI::gaspiCheckReturn(
      gaspi_notify_reset(local_segment_buf.get_segment_id(),
                         received_notification_id,
                         &received_notification_value),
      "PipelineCommunicator::blocking_recv : gaspi_notify_reset");
    if (received_notification_value != micro_id + 1)
    {
      throw std::runtime_error("PipelineCommunicator::blocking_recv : \
                                Incorrect notification value received");
    }

    copy_data_from_segment(local_recv_buf, local_segment_buf);
  }

  void PipelineCommunicator::copy_data_to_segment(void* local_send_buf,
                                                  GPI::SegmentBuffer const& segment_buffer)
  {
    auto const segment_ptr = segment_buffer.get_ptr();
    auto const buffer_size = segment_buffer.get_size();
    std::memcpy(segment_ptr, local_send_buf, buffer_size);
  }

  void PipelineCommunicator::copy_data_from_segment(void* local_recv_buf,
                                                    GPI::SegmentBuffer const& segment_buffer)
  {
    auto const segment_ptr = segment_buffer.get_ptr();
    auto const buffer_size = segment_buffer.get_size();
    std::memcpy(local_recv_buf, segment_ptr, buffer_size);
  }
}
