#pragma once

#include "allreduce.h"
#include "allreduceButterfly.h"
#include "collectives/BufferElementType.hpp"
#include "Operator.hpp"
#include "Resource.hpp"

namespace tarantella
{
  namespace collectives
  {
    namespace Allreduce
    {
      allreduce::dataType to_allreduce_dataType(const BufferElementType type);
      allreduce::reductionType to_allreduce_reductionType(
                                              const ReductionOp op);
      allreduceButterfly::segmentBuffer to_allreduce_segment_buffer(
                                              Resource const &resource);
    }
  }
}