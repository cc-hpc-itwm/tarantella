#include "TensorBroadcaster.hpp"

#include <GaspiCxx/collectives/non_blocking/collectives_lowlevel/BroadcastSendToAll.hpp>
#include <GaspiCxx/collectives/non_blocking/Broadcast.hpp>

#include <cstring>
#include <stdexcept>
#include <vector>

namespace tarantella
{
  TensorBroadcaster::TensorBroadcaster(std::vector<collectives::TensorInfo> const& tensor_infos,
                                       gaspi::group::Group const& group,
                                       gaspi::group::Rank root_rank)
  : rank(group.rank()),
    root(root_rank)
  {
    using T = float;
    auto const Algorithm = gaspi::collectives::BroadcastAlgorithm::SEND_TO_ALL;

    for (auto const& tensor_info : tensor_infos)
    {
      if (tensor_info.get_elem_type() != collectives::BufferElementType::FLOAT)
      {
        throw std::logic_error("[TensorBroadcaster] "
                               "only float supported");
      }

      auto const number_elements = tensor_info.get_nelems();
      broadcasts.push_back(std::make_unique<gaspi::collectives::Broadcast<T, Algorithm>>(
                           group, number_elements, root));
      sizes.push_back(number_elements);
    }
  }

  void TensorBroadcaster::exec_broadcast(std::vector<void const*> const& input_ptrs,
                                         std::vector<void*> const& output_ptrs)
  {
    if (input_ptrs.size() != broadcasts.size())
    {
      throw std::logic_error("[TensorBroadcaster::exec_broadcast] "
                             "number of inputs needs to stay the same");
    }
    if (input_ptrs.size() != output_ptrs.size())
    {
      throw std::logic_error("[TensorBroadcaster::exec_broadcast] "
                             "number of inputs and outputs have to be identical");
    }

    if (rank == root)
    {
      for (std::size_t i = 0; i < input_ptrs.size(); ++i)
      {
        broadcasts[i]->start(input_ptrs[i]);
      }
    }
    else
    {
      for (std::size_t i = 0; i < input_ptrs.size(); ++i)
      {
        broadcasts[i]->start();
      }
    }

    for (std::size_t i = 0; i < output_ptrs.size(); ++i)
    {
      broadcasts[i]->waitForCompletion(output_ptrs[i]);
    }
  }

  std::vector<std::size_t> TensorBroadcaster::get_sizes()
  {
    return sizes;
  }
}
