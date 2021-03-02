#include "BufferElementType.hpp"

#include <unordered_map>

namespace tarantella
{
  namespace collectives
  {
    std::size_t getDataTypeSize(const BufferElementType d)
    {
      std::unordered_map<BufferElementType, unsigned int> const sizes
      {
        {BufferElementType::FLOAT, sizeof(float)},
        {BufferElementType::DOUBLE, sizeof(double)},
        {BufferElementType::INT16, sizeof(int16_t)},
        {BufferElementType::INT32, sizeof(int32_t)}
      };
      return sizes.at(d);
    }

    std::ostream &operator<<(std::ostream& os, BufferElementType const& elem_type)
    {
      return os << static_cast<int>(elem_type);
    }
  }
}