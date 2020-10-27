#include "Segment.hpp"

namespace tarantella
{
  namespace GPI
  {
    Segment::Segment(Context& context,
                     Group const& group,
                     SegmentID id,
                     std::size_t size):
      context(context), group(group), id(id),
      size(size), ptr(nullptr)
    {
      context.allocate_segment(id, group, size);
      ptr = context.get_segment_pointer(id);
    }

    Segment::~Segment()
    {
      context.deallocate_segment(id, group);
    }

    SegmentID Segment::get_id() const
    {
      return id;
    }

    std::size_t Segment::get_size() const
    {
      return size;
    }

    void* Segment::get_ptr() const
    {
      return ptr;
    }
  }
}

