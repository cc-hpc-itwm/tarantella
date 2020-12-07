#pragma once

namespace tarantella
{
  namespace collectives
  {
    namespace Allreduce
    {
      enum class SegmentType
      {
        DATA1,
        DATA2,
        COMM
      };

      enum class ReductionOp
      {
        SUM,
        AVERAGE
      };
    }
  }
}