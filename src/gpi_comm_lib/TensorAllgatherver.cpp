#include "TensorAllgatherver.hpp"
#include "BufferElementType.hpp"

#include <GaspiCxx/collectives/non_blocking/collectives_lowlevel/AllgathervRing.hpp>

#include <cstring>
#include <numeric>
#include <algorithm> 
#include <stdexcept>
#include <vector>

namespace tarantella
{
  TensorAllgatherver::TensorAllgatherver(std::vector<collectives::TensorInfo> const& tensor_infos,
                                       gaspi::group::Group const& group)
  {
    using tensor_type = tarantella::collectives::BufferElementType;
    auto const Algorithm = gaspi::collectives::AllgathervAlgorithm::RING;

    for (auto const& tensor_info : tensor_infos)
    {
      switch(tensor_info.get_elem_type())
      {
        case tensor_type::FLOAT: 
            allgathervs.push_back(std::make_unique<gaspi::collectives::Allgatherv<float, Algorithm>>(
                          group, tensor_info.get_nelems()));
            break;
        case tensor_type::DOUBLE:
            allgathervs.push_back(std::make_unique<gaspi::collectives::Allgatherv<double, Algorithm>>(
                          group, tensor_info.get_nelems()));
            break;
        default: throw std::logic_error("TensorAllgatherver::TensorAllgatherver() Unsupported tensor data type");
      }
      auto counts = allgathervs.back() -> get_counts();
      sizes.push_back(std::accumulate(counts.begin(), counts.end(), 0));
    }
  }

  void TensorAllgatherver::exec_allgatherv(std::vector<void const*> const& input_ptrs,
                                         std::vector<void*> const& output_ptrs)
  {
    if (input_ptrs.size() != allgathervs.size())
    {
      throw std::logic_error("[TensorAllgatherver::exec_allgatherv] "
                             "number of inputs needs to stay the same");
    }
    if (input_ptrs.size() != output_ptrs.size())
    {
      throw std::logic_error("[TensorAllgatherver::exec_allgatherv] "
                             "number of inputs and outputs have to be identical");
    }

    for (std::size_t i = 0; i < input_ptrs.size(); ++i)
    {
      allgathervs[i]->start(input_ptrs[i]);
    }

    for (std::size_t i = 0; i < output_ptrs.size(); ++i)
    {
      allgathervs[i]->waitForCompletion(output_ptrs[i]);
    }
  }

  std::vector<std::size_t> TensorAllgatherver::get_output_sizes()
  {
    return sizes;
  }
}
