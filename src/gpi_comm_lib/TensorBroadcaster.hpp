#pragma once

#include "TensorInfo.hpp"

#include <GaspiCxx/collectives/non_blocking/Collective.hpp>
#include <GaspiCxx/group/Group.hpp>

#include <vector>

namespace tarantella
{
  using RootedSendCollective = gaspi::collectives::RootedSendCollective;

  class TensorBroadcaster
  {
    public:
      TensorBroadcaster(gaspi::group::Group const&,
                        std::vector<collectives::TensorInfo> const&,
                        gaspi::group::Rank root_rank);
      void exec_broadcast(std::vector<void*> const&);

    private:
      gaspi::group::Group const group;
      gaspi::group::Rank root;

      std::vector<collectives::TensorInfo> const tensor_infos;
      std::vector<char> bcast_buffer;
      std::unique_ptr<RootedSendCollective> bcast_op;
  };
}
