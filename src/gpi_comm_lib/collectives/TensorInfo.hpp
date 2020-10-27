#pragma once

#include "BufferElementType.hpp"
#include "Types.hpp"

#include <cstddef>

namespace tarantella
{
  namespace collectives
  {
    class TensorInfo
    {
      public:
        TensorInfo(GradID tensid, std::size_t nelems, BufferElementType elem_type);

        GradID get_id() const;
        std::size_t get_nelems() const;
        BufferElementType get_elem_type() const;
      
      private:
        const GradID id;
        const std::size_t nelems;
        const BufferElementType elem_type;
    };
  }
}
