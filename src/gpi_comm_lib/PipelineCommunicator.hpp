#pragma once

#include <GaspiCxx/group/Group.hpp>
#include <GaspiCxx/singlesided/write/SourceBuffer.hpp>
#include <GaspiCxx/singlesided/write/TargetBuffer.hpp>

#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tarantella
{
  class PipelineCommunicator
  {
    public:
      using ConnectionID = std::size_t;
      using MicrobatchID = std::size_t;
      using PartnerRank = gaspi::group::GlobalRank;
      using MessageSizeBytes = std::size_t;
      using LayerEdges = std::unordered_map<ConnectionID,
                                            std::pair<PartnerRank, MessageSizeBytes>>;

      using SourceBuffer = gaspi::singlesided::write::SourceBuffer;
      using TargetBuffer = gaspi::singlesided::write::TargetBuffer;

      PipelineCommunicator(LayerEdges const& edges,
                           std::size_t num_micro_batches);

      void send(void* local_send_buf, ConnectionID, MicrobatchID);
      void recv(void* local_recv_buf, ConnectionID, MicrobatchID);

    private:
      std::size_t num_micro_batches;
      std::unordered_map<ConnectionID, std::vector<std::unique_ptr<SourceBuffer>>> send_buffers;
      std::unordered_map<ConnectionID, std::vector<std::unique_ptr<TargetBuffer>>> receive_buffers;
  };
}
