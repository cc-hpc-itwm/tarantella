#include "PipelineCommunicator.hpp"

#include <GaspiCxx/Runtime.hpp>
#include <GaspiCxx/singlesided/Endpoint.hpp>

#include <algorithm>
#include <cstring>

namespace tarantella
{
  PipelineCommunicator::PipelineCommunicator(
    LayerEdges const& edges,
    std::size_t num_micro_batches)
  : num_micro_batches(num_micro_batches)
  {
    for (auto& [connection_id, edge] : edges)
    {
      auto size_bytes = edge.second;

      std::vector<std::unique_ptr<SourceBuffer>> send_buffers_micro_batches(num_micro_batches);
      std::generate(send_buffers_micro_batches.begin(), send_buffers_micro_batches.end(),
                    [size_bytes]()
                    {
                      return std::make_unique<SourceBuffer>(size_bytes);
                    });
      send_buffers.emplace(connection_id, std::move(send_buffers_micro_batches));

      std::vector<std::unique_ptr<TargetBuffer>> recv_buffers_micro_batches(num_micro_batches);
      std::generate(recv_buffers_micro_batches.begin(), recv_buffers_micro_batches.end(),
                    [size_bytes]()
                    {
                      return std::make_unique<TargetBuffer>(size_bytes);
                    });
      receive_buffers.emplace(connection_id, std::move(recv_buffers_micro_batches));
    }

    using ConnectHandle = gaspi::singlesided::Endpoint::ConnectHandle;
    std::vector<ConnectHandle> connection_handles;
    for (auto& [connection_id, buffers] : send_buffers)
    {
      for (auto micro_batch_id = 0UL; micro_batch_id < num_micro_batches; ++micro_batch_id)
      {
        SourceBuffer::Tag const tag = connection_id * num_micro_batches + micro_batch_id;
        auto partner_rank = edges.at(connection_id).first;
        gaspi::group::Group group({gaspi::getRuntime().global_rank(), partner_rank});

        auto handle = buffers[micro_batch_id]->connectToRemoteTarget(
                                                group,
                                                group.toGroupRank(partner_rank),
                                                tag);
        connection_handles.emplace_back(std::move(handle));
      }
    }

    for (auto& [connection_id, buffers] : receive_buffers)
    {
      for (auto micro_batch_id = 0UL; micro_batch_id < num_micro_batches; ++micro_batch_id)
      {
        TargetBuffer::Tag const tag = connection_id * num_micro_batches + micro_batch_id;
        auto partner_rank = edges.at(connection_id).first;
        gaspi::group::Group group({gaspi::getRuntime().global_rank(), partner_rank});
        auto handle = buffers[micro_batch_id]->connectToRemoteSource(
                                                group,
                                                group.toGroupRank(partner_rank),
                                                tag);
        connection_handles.push_back(std::move(handle));
      }
    }

    for (auto& handle : connection_handles)
    {
      handle.waitForCompletion();
    }
  }

  void PipelineCommunicator::send(void* const local_send_buf,
                                               ConnectionID conn_id,
                                               MicrobatchID micro_id)
  {
    auto& buffer = send_buffers[conn_id][micro_id];
    auto buffer_ptr = buffer->address();
    std::memcpy(buffer_ptr, local_send_buf, buffer->description().size());

    buffer->initTransfer();
    if (micro_id == num_micro_batches - 1)
    {
      // block in send until the data is received for the last micro-batch to ensure
      // mini-batch-level synchronization between partitions (within the forward pass)
      buffer->waitForTransferAck();
    }
  }

  void PipelineCommunicator::recv(void* local_recv_buf,
                                           ConnectionID conn_id,
                                           MicrobatchID micro_id)
  {
    auto& buffer = receive_buffers[conn_id][micro_id];
    buffer->waitForCompletion();
    auto const buffer_ptr = buffer->address();
    std::memcpy(local_recv_buf, buffer_ptr, buffer->description().size());

    if (micro_id == num_micro_batches - 1)
    {
      // acknowledge data transfer after the last micro-batch to ensure
      // mini-batch-level synchronization between partitions (within the forward pass)
      buffer->ackTransfer();
    }
  }
}
