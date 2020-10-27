#include "SegmentIDBuilder.hpp"

namespace tarantella
{
  namespace distribution
  {
    GPI::SegmentID DataParallelSegmentIDBuilder::segment_id = 0UL;

    GPI::SegmentID DataParallelSegmentIDBuilder::get_segment_id()
    {
      return segment_id++;
    }

    GPI::SegmentID PipelineSegmentIDBuilder::get_segment_id(PipelineCommunicator::ConnectionID id)
    {
      return static_cast<GPI::SegmentID>(id);
    }
  }
}