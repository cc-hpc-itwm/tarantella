#pragma once

#include "TensorInfo.hpp"

#include <GaspiCxx/collectives/non_blocking/Gatherv.hpp>
#include <GaspiCxx/collectives/non_blocking/Collective.hpp>
#include <GaspiCxx/group/Group.hpp>

#include <vector>
namespace tarantella
{
  class TensorGatherver
  {
    public:
      TensorGatherver(std::vector<collectives::TensorInfo> const &,
                        gaspi::group::Group const &,
                        gaspi::group::Rank root_rank);
      void exec_gatherv(std::vector<void const*> const& input_ptrs,
                          std::vector<void*> const& output_ptrs);
      void exec_gatherv(std::vector<void const*> const& input_ptrs);
      gaspi::group::Rank get_root();
      std::vector<std::size_t> get_output_sizes();
      
    private:
      gaspi::group::Rank rank;
      gaspi::group::Rank root;
      std::vector<std::unique_ptr<gaspi::collectives::VariableSizeInputCollective>> gathervs;
      std::vector<std::size_t> sizes;
  };
}