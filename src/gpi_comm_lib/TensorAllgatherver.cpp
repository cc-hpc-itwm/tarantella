#include "TensorAllgatherver.hpp"

#include <stdexcept>

namespace tarantella
{
  TensorAllgatherver::TensorAllgatherver(std::size_t nelems, TensorType dtype,
                                         gaspi::group::Group const& group)
  {
    auto const Algorithm = gaspi::collectives::AllgathervAlgorithm::RING;

    switch(dtype)
    {
      case TensorType::FLOAT:
        allgatherv = std::make_unique<gaspi::collectives::Allgatherv<float, Algorithm>>(group, nelems);
        break;
      case TensorType::DOUBLE:
        allgatherv = std::make_unique<gaspi::collectives::Allgatherv<double, Algorithm>>(group, nelems);
        break;
      default:
        throw std::logic_error("TensorAllgatherver::TensorAllgatherver() Unsupported tensor data type");
    }
    size = allgatherv->getOutputCount();
  }

  void TensorAllgatherver::exec_allgatherv(const void* input_ptr,
                                           void* output_ptr)
  {
    allgatherv->start(input_ptr);
    allgatherv->waitForCompletion(output_ptr);
  }

  std::size_t TensorAllgatherver::getOutputCount()
  {
    return allgatherv->getOutputCount();
  }
}
