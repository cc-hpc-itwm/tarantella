#include "counter.h"

namespace tarantella
{
  namespace collectives
  {
    counter::counter(const unsigned long phasePeriod_)
    :  phasePeriod(phasePeriod_),
       value(0) {}
    
    unsigned long counter::increment() {
      return (++value) % phasePeriod;
    }
    
    unsigned long counter::get() const {
      return value % phasePeriod;
    }
  }
}
