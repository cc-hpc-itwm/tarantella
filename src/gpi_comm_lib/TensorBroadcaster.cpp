#include "TensorBroadcaster.hpp"

#include <GaspiCxx/collectives/non_blocking/collectives_lowlevel/BroadcastSendToAll.hpp>
#include <GaspiCxx/collectives/non_blocking/Broadcast.hpp>

#include <cstring>
#include <vector>

namespace tarantella
{
  TensorBroadcaster::TensorBroadcaster(std::vector<collectives::TensorInfo> const& tensor_infos,
                                       gaspi::group::Group const& group,
                                       gaspi::group::Rank root_rank)
  : rank(group.rank()),
    root(root_rank)
  {
    auto const Algorithm = gaspi::collectives::BroadcastAlgorithm::SEND_TO_ALL;

    for (auto const& tensor_info : tensor_infos)
    {
      broadcasts.push_back(std::make_unique<gaspi::collectives::Broadcast<char, Algorithm>>(
                           group, tensor_info.get_size_bytes(), root));

    }
  }

  void TensorBroadcaster::exec_broadcast(std::vector<void*> const& data_ptrs)
  {
    if (data_ptrs.size() != broadcasts.size())
    {
      throw std::logic_error("[TensorBroadcaster::exec_broadcast] "
                             "number of inputs needs to stay the same");
    }

    if (rank == root)
    {
      for (std::size_t i = 0; i < data_ptrs.size(); ++i)
      {
        broadcasts[i]->start(data_ptrs[i]);
      }
    }
    else
    {
      for (std::size_t i = 0; i < data_ptrs.size(); ++i)
      {
        broadcasts[i]->start();
      }
    }

    for (std::size_t i = 0; i < data_ptrs.size(); ++i)
    {
      broadcasts[i]->waitForCompletion(data_ptrs[i]);
    }
  }
}

