#include "TensorInfo.hpp"

namespace tarantella
{
  namespace collectives
  {
    TensorInfo::TensorInfo(GradID tensid, std::size_t nelems, BufferElementType elem_type)
    : id(tensid), nelems(nelems), elem_type(elem_type)
    {}

    GradID TensorInfo::get_id() const
    {
      return id;
    }

    std::size_t TensorInfo::get_nelems() const
    {
      return nelems;
    }

    BufferElementType TensorInfo::get_elem_type() const
    {
      return elem_type;
    }
  }
}
