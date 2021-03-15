#include "PipelineCommunicator.hpp"

namespace tarantella
{
  PipelineCommunicator::PipelineCommunicator(
    LayerEdges const&, // connection_infos,
    std::size_t) // num_micro_batches)
  { }

  void PipelineCommunicator::non_blocking_send(void*, // local_send_buf,
                                               ConnectionID, // conn_id,
                                               MicrobatchID) // micro_id)
  { }

  void PipelineCommunicator::blocking_recv(void*, //local_recv_buf,
                                           ConnectionID, // conn_id,
                                           MicrobatchID) // micro_id)
  {  }

}
