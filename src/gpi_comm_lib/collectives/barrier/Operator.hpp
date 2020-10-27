#pragma once

namespace tarantella
{
  namespace collectives
  {
    namespace Barrier
    {
      // \note
      // Interface for Barrier algorithms (not thread-safe)
      class Operator
      {
      public:
        virtual ~Operator() = default;

        virtual void blocking_barrier() = 0;
      };
    } 
  }   
} 
