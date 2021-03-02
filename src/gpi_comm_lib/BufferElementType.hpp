#pragma once

#include <cstddef>
#include <iostream>

namespace tarantella
{
  namespace collectives
  {
    enum class BufferElementType
    {
      FLOAT,
      DOUBLE,
      INT16,
      INT32
    };

    std::size_t getDataTypeSize(const BufferElementType d);
    std::ostream &operator<<(std::ostream& os, BufferElementType const& elem_type);
  }
}
