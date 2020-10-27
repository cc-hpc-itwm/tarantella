#pragma once

#include "Context.hpp"
#include "Segment.hpp"
#include "SegmentBuffer.hpp"

#include <cstddef>
#include <memory>
#include <unordered_map>

namespace tarantella
{
  namespace GPI
  {
    class SegmentManager
    {
      public:
        SegmentManager(GPI::Context&);
        SegmentManager() = delete;
        SegmentManager(SegmentManager const&) = delete;
        SegmentManager& operator=(SegmentManager const&) = delete;
        ~SegmentManager() = default;

        void create_segment(GPI::SegmentID, GPI::Group const&, std::size_t);
        SegmentBuffer get_buffer_of_size(GPI::SegmentID, std::size_t);

      private:
        class AllocatedSegment
        {
          public:
            AllocatedSegment(GPI::Context& context, GPI::Group const& group, 
                             GPI::SegmentID id, std::size_t size, std::size_t offset)
            : segment(std::make_unique<GPI::Segment>(context, group, id, size)),
              current_offset(offset)
            {}
            AllocatedSegment(AllocatedSegment&&) = default;
            AllocatedSegment& operator=(AllocatedSegment&&) = default;

            std::unique_ptr<GPI::Segment> segment;
            std::size_t current_offset;
        };

        GPI::Context& context;
        std::unordered_map<GPI::SegmentID, AllocatedSegment> segments;
    };
  }
}
