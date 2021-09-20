#pragma once

#include "TensorInfo.hpp"

#include <GaspiCxx/collectives/non_blocking/Allgatherv.hpp>
#include <GaspiCxx/collectives/non_blocking/Collective.hpp>
#include <GaspiCxx/group/Group.hpp>

#include <vector>

namespace tarantella
{
  class TensorAllgatherver
  {
    public:
      TensorAllgatherver(std::vector<collectives::TensorInfo> const &,
                        gaspi::group::Group const &);
      void exec_allgatherv(std::vector<void const*> const& input_ptrs,
                          std::vector<void*> const& output_ptrs);
      std::vector<std::size_t> get_output_sizes();
    private:
      std::vector<std::unique_ptr<gaspi::collectives::VariableSizeInputCollective>> allgathervs;
      std::vector<std::size_t> sizes;
  };
}
