#pragma once

#include "PipelineCommunicator.hpp"

namespace tarantella
{
  namespace distribution
  {
    class DataParallelSegmentIDBuilder
    {
      public:
        DataParallelSegmentIDBuilder() = default;

        GPI::SegmentID get_segment_id();

      private:
        static GPI::SegmentID segment_id;
    };

    class PipelineSegmentIDBuilder
    {
      public:
        PipelineSegmentIDBuilder() = default;

        GPI::SegmentID get_segment_id(PipelineCommunicator::ConnectionID id);
    };
  }
}