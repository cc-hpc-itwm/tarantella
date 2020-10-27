#pragma once

namespace tarantella
{
  namespace collectives
  {
    class allreduce {
    public:
      enum reductionType {
        SUM = 0,
        AVERAGE = 1,
        NUM_RED = 2
      };
      enum dataType {
        FLOAT = 0,
        DOUBLE = 1,
        INT16 = 2,
        INT32 = 3,
        NUM_TYPE = 4
      };

      virtual int operator()() = 0;
      virtual void signal() = 0;
      virtual ~allreduce() {}
    };
  }
}
