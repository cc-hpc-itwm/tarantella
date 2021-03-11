#include "PipelineCommunicator.hpp"

#include <cstring>

namespace tarantella
{
  PipelineCommunicator::PipelineCommunicator(
    LayerEdges const& edges,
    std::size_t num_micro_batches)
  {
    for (auto& [connection_id, edge] : edges)
    {
      auto partner_rank = edge.first;
      auto size_bytes = edge.second;

      auto send_buffers_micro_batches = std::vector<std::unique_ptr<SourceBuffer>>(
                                                    num_micro_batches,
                                                    std::make_unique<SourceBuffer>(size_bytes));
      send_buffers.emplace(connection_id, send_buffers_micro_batches);

      auto recv_buffers_micro_batches = std::vector<std::unique_ptr<TargetBuffer>>(
                                                    num_micro_batches,
                                                    std::make_unique<TargetBuffer>(size_bytes));
      receive_buffers.emplace(connection_id, recv_buffers_micro_batches);
    }

    for (auto& [connection_id, buffers] : send_buffers)
    {
      for (auto micro_batch_id = 0UL; micro_batch_id < num_micro_batches; ++micro_batch_id)
      {
        SourceBuffer::Tag const tag = connection_id * num_micro_batches + micro_batch_id;
        auto partner_rank = edges[connection_id].first;
        gaspi::group::Group group({gaspi::getRuntime().globalRank(), partner_rank});
        buffer->connectToRemoteTarget(group, group.toGroupRank(partner_rank), tag);
      }
    }

    for (auto& [connection_id, buffers] : receive_buffers)
    {
      for (auto micro_batch_id = 0UL; micro_batch_id < num_micro_batches; ++micro_batch_id)
      {
        TargetBuffer::Tag const tag = connection_id * num_micro_batches + micro_batch_id;
        auto partner_rank = edges[connection_id].first;
        gaspi::group::Group group({gaspi::getRuntime().globalRank(), partner_rank});
        buffers[micro_batch_id]->connectToRemoteSource(group, group.toGroupRank(partner_rank), tag);
      }
    }

    for (auto& [_, buffers] : send_buffers)
    {
      for (auto& buffer : buffers)
      {
        buffer->waitForCompletion();
      }
    }
    for (auto& [_, buffers] : receive_buffers)
    {
      for (auto& buffer : buffers)
      {
        buffer->waitForCompletion();
      }
    }
  }

  void PipelineCommunicator::non_blocking_send(void* const local_send_buf,
                                               ConnectionID conn_id,
                                               MicrobatchID micro_id)
  {
    auto& buffer = send_buffers[conn_id][micro_id];
    auto buffer_ptr = buffer->address();
    std::memcpy(buffer_ptr, local_send_buf, buffer->description().size());

    buffer->initTransfer();
  }

  void PipelineCommunicator::blocking_recv(void* local_recv_buf,
                                           ConnectionID conn_id,
                                           MicrobatchID micro_id)
  {
    auto& buffer = receive_buffers[conn_id][micro_id];
    buffer->waitForCompletion();

    auto const buffer_ptr = buffer->address();
    std::memcpy(local_recv_buf, buffer_ptr, buffer->description().size());
  }
}
