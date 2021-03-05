#include "TensorAllreducer.hpp"

#include <GaspiCxx/collectives/non_blocking/collectives_lowlevel/AllreduceRing.hpp>

#include <cstring>
#include <stdexcept>

namespace tarantella
{
  TensorAllreducer::TensorAllreducer(std::vector<TensorInfo> const& tensor_infos,
                                     gaspi::group::Group const& group,
                                     gaspi::collectives::ReductionOp const& reduction_op)
  {
    using T = float;
    auto const Algorithm = gaspi::collectives::AllreduceAlgorithm::RING;

    for (auto const& tensor_info : tensor_infos)
    {
      allreduces.push_back(std::make_unique<gaspi::collectives::Allreduce<T, Algorithm>>(
                           group, tensor_info.get_nelems(), reduction_op));

    }
  }

  void TensorAllreducer::exec_allreduce(std::vector<const void*> const& input_ptrs,
                                        std::vector<void*> const& output_ptrs)
  {
    if (input_ptrs.size() != allreduces.size())
    {
      throw std::logic_error("[TensorAllreducer::exec_allreduce] "
                             "number of inputs needs to stay the same");
    }
    if (input_ptrs.size() != output_ptrs.size())
    {
      throw std::logic_error("[TensorAllreducer::exec_allreduce] "
                             "number of inputs and outputs have to be identical");
    }

    for (std::size_t i = 0; i < input_ptrs.size(); ++i)
    {
      allreduces[i]->start(input_ptrs[i]);
    }

    for (std::size_t i = 0; i < input_ptrs.size(); ++i)
    {
      allreduces[i]->waitForCompletion(output_ptrs[i]);
    }
  }
}