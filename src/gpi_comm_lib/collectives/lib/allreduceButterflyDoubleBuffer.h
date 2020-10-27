#pragma once

#include "allreduceButterfly.h"
#include "gpi/Group.hpp"

namespace tarantella
{
  namespace collectives
  {
    class allreduceButterflyDoubleBuffer : public allreduce {
    public:
    
      allreduceButterflyDoubleBuffer(
        const long len,
        const dataType data,
        const reductionType reduction,
        const allreduceButterfly::segmentBuffer segmentReduce0,
        const allreduceButterfly::segmentBuffer segmentReduce1,
        const allreduceButterfly::segmentBuffer segmentCommunicate,
        queues& queues,
        GPI::Group const& group);
      int operator()();
      void signal();
    
      gaspi_pointer_t getActiveReducePointer() const;
      gaspi_pointer_t getResultsPointer() const;
      static long getNumberOfElementsSegmentCommunicate(const long len,
                                                        const long numRanks);
      static unsigned long getNumberOfNotifications(const long numRanks);
      std::ostream& report(std::ostream& s) const;
    
    private:
    
      inline allreduceButterfly& getReduce() const;
      static inline long stateToIndex(const long state);
      inline void flipReduce();
      inline const allreduceButterfly& getOtherReduce() const;
      static inline long invertIndex(const long state);
    
      static const long CACHE_LINE_SIZE = 64;
    
      char pad0[CACHE_LINE_SIZE];
      volatile long state;
      char pad1[CACHE_LINE_SIZE];
    
      allreduceButterfly reduceFirst;
      allreduceButterfly reduceSecond;
      allreduceButterfly* tableReduce[2];
    };
  }
}
    