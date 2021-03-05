#pragma once

#include "TensorInfo.hpp"

#include <GaspiCxx/collectives/non_blocking/Allreduce.hpp>
#include <GaspiCxx/collectives/non_blocking/Collective.hpp>
#include <GaspiCxx/group/Group.hpp>

#include <memory>
#include <vector>

namespace tarantella
{
  class TensorAllreducer
  {
    public:
      using TensorInfo = tarantella::collectives::TensorInfo;

      TensorAllreducer(std::vector<TensorInfo> const&,
                       gaspi::group::Group const&,
                       gaspi::collectives::ReductionOp const&);
      void exec_allreduce(std::vector<const void*> input_ptrs,
                          std::vector<void*> output_ptrs);

    private:
      std::vector<std::unique_ptr<gaspi::collectives::Collective>> allreduces;
  };
}