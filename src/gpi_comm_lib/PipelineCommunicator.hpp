#pragma once

#include <GaspiCxx/group/Group.hpp>

#include <unordered_map>
#include <utility>

namespace tarantella
{
  class PipelineCommunicator
  {
    public:
      using ConnectionID = std::size_t;
      using MicrobatchID = std::size_t;
      using LayerEdges = std::unordered_map<tarantella::PipelineCommunicator::ConnectionID,
                                            std::pair<std::pair<gaspi::group::GlobalRank, 
                                                                gaspi::group::GlobalRank>, std::size_t>>;

      PipelineCommunicator(LayerEdges const&,
                           std::size_t num_micro_batches);

      void non_blocking_send(void* local_send_buf,
                             ConnectionID,
                             MicrobatchID);
      void blocking_recv(void* local_recv_buf,
                         ConnectionID,
                         MicrobatchID);
  };
}
