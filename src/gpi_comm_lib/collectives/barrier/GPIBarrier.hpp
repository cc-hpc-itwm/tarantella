#pragma once

#include "gpi/Group.hpp"
#include "Operator.hpp"

namespace tarantella
{
  namespace collectives
  {
    namespace Barrier
    {
      // GPI Barrier implementation for GROUP_COMM_ALL
      class GPIBarrier : public Operator
      {
        public:
        
          GPIBarrier(GPI::Group const & group);
          void blocking_barrier();
      };

      class GPIBarrierAllRanks : public Operator
      {
        public:

          GPIBarrierAllRanks() = default;
          void blocking_barrier();
      };
    } 
  }   
} 
