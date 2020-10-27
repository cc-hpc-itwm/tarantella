
#include "SegmentBuffer.hpp"

#include <cstddef>

namespace tarantella
{
  namespace GPI
  {
    SegmentBuffer::SegmentBuffer(GPI::Segment const& s, std::size_t offset, std::size_t size)
    : id(s.get_id()), offset(offset), size(size),
      ptr(reinterpret_cast<std::byte*>(s.get_ptr()) + offset)
    { }

    SegmentID SegmentBuffer::get_segment_id() const { return id; }
    std::size_t SegmentBuffer::get_size() const { return size; }
    std::size_t SegmentBuffer::get_offset() const { return offset; }
    void* SegmentBuffer::get_ptr() const { return ptr; }

  }
}