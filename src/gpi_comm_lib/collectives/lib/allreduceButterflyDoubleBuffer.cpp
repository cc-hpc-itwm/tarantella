#include "allreduceButterflyDoubleBuffer.h"

namespace tarantella
{
  namespace collectives
  {
    allreduceButterflyDoubleBuffer::allreduceButterflyDoubleBuffer(
      const long len,
      const dataType data,
      const reductionType reduction,
      const allreduceButterfly::segmentBuffer segmentReduce0,
      const allreduceButterfly::segmentBuffer segmentReduce1,
      const allreduceButterfly::segmentBuffer segmentCommunicate,
      queues& queues,
      GPI::Group const& group)
    : state(0),
      reduceFirst(len, data, reduction, segmentReduce0,
                  segmentCommunicate, queues, group),
      reduceSecond(len, data, reduction, segmentReduce1,
                   segmentCommunicate, queues, group) {
      tableReduce[0] = &reduceFirst;
      tableReduce[1] = &reduceSecond;
    }
    
    int allreduceButterflyDoubleBuffer::operator()() {
      const int result = getReduce()();
    
      if (!result) {
        flipReduce();
      }
    
      return result;
    }
    
    inline allreduceButterfly& allreduceButterflyDoubleBuffer::getReduce() const {
      return tableReduce[stateToIndex(state)][0];
    }
    
    inline long allreduceButterflyDoubleBuffer::stateToIndex(const long state) {
      return state & 1l;
    }
    
    inline void allreduceButterflyDoubleBuffer::flipReduce() {
      __sync_fetch_and_add(&state, 1l);
    }
    
    void allreduceButterflyDoubleBuffer::signal() {
      getReduce().signal();
    }
    
    gaspi_pointer_t allreduceButterflyDoubleBuffer::getActiveReducePointer() const {
      return getReduce().getReducePointer();
    }
    
    gaspi_pointer_t allreduceButterflyDoubleBuffer::getResultsPointer() const {
      return getOtherReduce().getReducePointer();
    }
    
    inline const allreduceButterfly&
      allreduceButterflyDoubleBuffer::getOtherReduce() const {
      return tableReduce[invertIndex(stateToIndex(state))][0];
    }
    
    inline long allreduceButterflyDoubleBuffer::invertIndex(const long state) {
      return state ^ 1l;
    }
    
    long allreduceButterflyDoubleBuffer::getNumberOfElementsSegmentCommunicate(
      const long len,
      const long numRanks) {
      return allreduceButterfly::getNumberOfElementsSegmentCommunicate(len,
                                                                      numRanks);
    }
    
    unsigned long allreduceButterflyDoubleBuffer::getNumberOfNotifications(
      const long numRanks) {
      return allreduceButterfly::getNumberOfNotifications(numRanks);
    }
    
    std::ostream& allreduceButterflyDoubleBuffer::report(std::ostream& s) const {
      s << "stateExecute: " << state << std::endl
        << "***** reduceFirst *****" << std::endl;
      reduceFirst.report(s);
      s << "***** reduceSecond *****" << std::endl;
      reduceSecond.report(s);
    
      return s;
    }
  }
}
    