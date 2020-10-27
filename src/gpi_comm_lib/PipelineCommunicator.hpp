#pragma once

#include <gpi/Context.hpp>
#include <gpi/ResourceManager.hpp>
#include <gpi/SegmentBuffer.hpp>

#include <unordered_map>
#include <utility>

namespace tarantella
{
  class SendRecvResources
  {
    public:
      SendRecvResources() = default;
      SendRecvResources(GPI::Rank rank,
                        std::vector<GPI::SegmentBuffer> const& send_bufs,
                        std::vector<GPI::SegmentBuffer> const& recv_bufs,
                        std::vector<GPI::NotificationManager::NotificationID> const& notifications)
      : other_rank(rank), send_bufs(send_bufs), recv_bufs(recv_bufs), notifications(notifications)
      {}

      GPI::Rank other_rank;
      std::vector<GPI::SegmentBuffer> send_bufs;
      std::vector<GPI::SegmentBuffer> recv_bufs;
      std::vector<GPI::NotificationManager::NotificationID> notifications;
  };

  class ConnectionInfo
  {
    public:
      explicit ConnectionInfo(GPI::SegmentID segment_id, GPI::Rank other_rank, std::size_t buffer_size_bytes)
      : segment_id(segment_id), other_rank(other_rank), microbatched_buffer_size_bytes(buffer_size_bytes)
      {}

      GPI::SegmentID segment_id;
      GPI::Rank other_rank;
      std::size_t microbatched_buffer_size_bytes;
  };

  class PipelineCommunicator
  {
    public:
      using ConnectionID = std::size_t;
      using MicrobatchID = std::size_t;

      PipelineCommunicator(GPI::Context&,
                           std::unordered_map<ConnectionID, ConnectionInfo> const&,
                           std::size_t num_micro_batches);

      void non_blocking_send(void* local_send_buf,
                             ConnectionID,
                             MicrobatchID);
      void blocking_recv(void* local_recv_buf,
                         ConnectionID,
                         MicrobatchID);

    private:
      GPI::ResourceManager& resource_manager;
      std::unordered_map<ConnectionID, SendRecvResources> connections;

      void copy_data_to_segment(void* local_send_buf, GPI::SegmentBuffer const&);
      void copy_data_from_segment(void* local_recv_buf, GPI::SegmentBuffer const&);
  };
}
