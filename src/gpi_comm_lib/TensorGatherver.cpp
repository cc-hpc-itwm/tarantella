#include "TensorGatherver.hpp"
#include "BufferElementType.hpp"

#include <GaspiCxx/collectives/non_blocking/collectives_lowlevel/GathervBinomial.hpp>

#include <cstring>
#include <numeric>
#include <algorithm> 
#include <stdexcept>
#include <vector>

namespace tarantella
{
  TensorGatherver::TensorGatherver(std::vector<collectives::TensorInfo> const& tensor_infos,
                                       gaspi::group::Group const& group,
                                       gaspi::group::Rank root_rank)
  : rank(group.rank()),
    root(root_rank)
  {
    using tensor_type = tarantella::collectives::BufferElementType;
    auto const Algorithm = gaspi::collectives::GathervAlgorithm::BINOMIAL;

    for (auto const& tensor_info : tensor_infos)
    {
      switch(tensor_info.get_elem_type())
      {
        case tensor_type::FLOAT: 
            gathervs.push_back(std::make_unique<gaspi::collectives::Gatherv<float, Algorithm>>(
                          group, root_rank, tensor_info.get_nelems()));
            break;
        case tensor_type::DOUBLE:
            gathervs.push_back(std::make_unique<gaspi::collectives::Gatherv<double, Algorithm>>(
                          group, root_rank, tensor_info.get_nelems()));
            break;
        default: throw std::logic_error("TensorGatherver::TensorGatherver() Unsupported tensor data type");
      }
      auto counts = gathervs.back() -> get_counts();
      sizes.push_back(std::accumulate(counts.begin(), counts.end(), 0));
    }
  }

  void TensorGatherver::exec_gatherv(std::vector<void const*> const& input_ptrs,
                      std::vector<void*> const& output_ptrs)
  {
    if(rank != root)
    {
      throw std::logic_error("[TensorAllgatherver::exec_gatherv] "
                             "only root rank get output");
    }
    if (input_ptrs.size() != gathervs.size())
    {
      throw std::logic_error("[Tensorgatherver::exec_gatherv] "
                             "number of inputs needs to stay the same");
    }
    if (input_ptrs.size() != output_ptrs.size())
    {
      throw std::logic_error("[Tensorgatherver::exec_gatherv] "
                             "number of inputs and outputs have to be identical");
    }

    for (std::size_t i = 0; i < input_ptrs.size(); ++i)
    {
      gathervs[i]->start(input_ptrs[i]);
    }

    for (std::size_t i = 0; i < output_ptrs.size(); ++i)
    {
      gathervs[i]->waitForCompletion(output_ptrs[i]);
    }
  }

  void TensorGatherver::exec_gatherv(std::vector<void const*> const& input_ptrs)
  {
    if(rank == root)
    {
      throw std::logic_error("[TensorAllgatherver::exec_gatherv] "
                             "root rank should get output");
    }
    if (input_ptrs.size() != gathervs.size())
    {
      throw std::logic_error("[Tensorgatherver::exec_gatherv] "
                             "number of inputs needs to stay the same");
    }

    for (std::size_t i = 0; i < input_ptrs.size(); ++i)
    {
      gathervs[i]->start(input_ptrs[i]);
    }

    for (std::size_t i = 0; i < input_ptrs.size(); ++i)
    {
      gathervs[i]->waitForCompletion(nullptr);
    }
  }

  std::vector<std::size_t> TensorGatherver::get_output_sizes()
  {
    return sizes;
  }

  gaspi::group::Rank TensorGatherver::get_root()
  {
    return root;
  }
}