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
      using TensorType = tarantella::collectives::BufferElementType;

      TensorAllgatherver(std::size_t nelems, TensorType dtype,
                        gaspi::group::Group const &);
      void exec_allgatherv(void const* input_ptr,
                           void* output_ptr);
      std::size_t getOutputCount();
    private:
      std::unique_ptr<gaspi::collectives::Collective> allgatherv;
      std::size_t size;
  };
}
