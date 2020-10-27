#pragma once

#include<atomic>

namespace tarantella
{
  namespace collectives
  {
    class counter {
    public:
      counter(const unsigned long phasePeriod_ = 1);
      unsigned long increment();
      unsigned long get() const;
    private:
    
      const unsigned long phasePeriod;
      std::atomic<unsigned long> value;
    };
  }
}
    