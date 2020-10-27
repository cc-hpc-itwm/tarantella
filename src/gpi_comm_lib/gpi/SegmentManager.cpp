#include "SegmentManager.hpp"

#include <memory>
#include <stdexcept>

namespace tarantella
{
  namespace GPI
  {
    SegmentManager::SegmentManager(GPI::Context& context)
    : context(context), segments()
    { }

    void SegmentManager::create_segment(GPI::SegmentID id, GPI::Group const& group, std::size_t size)
    {
      if(segments.find(id) != segments.end())
      {
        throw std::runtime_error("[SegmentManager::create_segment]:\
                                  Segment already exists");
      }
      segments.emplace(std::make_pair(id, AllocatedSegment(context, group, id, size, 0UL)));
    }

    SegmentBuffer SegmentManager::get_buffer_of_size(GPI::SegmentID id, std::size_t buffer_size)
    {
      if(segments.find(id) == segments.end())
      {
        throw std::runtime_error("[SegmentManager::get_buffer_of_size]:\
                                  Segment not allocated");
      }

      auto& segment = segments.at(id).segment;
      auto const current_offset = segments.at(id).current_offset;
      if(current_offset + buffer_size > segment->get_size())
      {
        throw std::runtime_error("[SegmentManager::get_buffer_of_size]:\
                                  Out of memory");
      }

      SegmentBuffer const segmentBuffer(*segment, current_offset, buffer_size);
      segments.at(id).current_offset = current_offset + buffer_size;
      return segmentBuffer;
    }
  }
}
