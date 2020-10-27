#pragma once

#include "Segment.hpp"

namespace tarantella
{
  namespace GPI
  {
    class SegmentBuffer
    {
      public:
        SegmentBuffer(GPI::Segment const &s, std::size_t offset, std::size_t size);
        SegmentBuffer(SegmentBuffer const& other) = default;
        SegmentBuffer& operator=(SegmentBuffer const&) = delete;
        SegmentBuffer(SegmentBuffer&&) = default;
        SegmentBuffer& operator=(SegmentBuffer&&) = delete;
        ~SegmentBuffer() = default;

        SegmentID get_segment_id() const;
        std::size_t get_size() const;
        std::size_t get_offset() const;
        void* get_ptr() const;

      private:
        SegmentID const id;
        std::size_t const offset;
        std::size_t const size;
        void* const ptr;
    };
  }
}