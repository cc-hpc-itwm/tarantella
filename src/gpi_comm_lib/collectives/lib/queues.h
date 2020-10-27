#pragma once

#include <GASPI.h>
#include <vector>

namespace tarantella
{
  namespace collectives
  {
    class queues {
    public:
      queues(const long num = 2,
            const gaspi_queue_id_t first = 0);
      queues(const std::vector<gaspi_queue_id_t>& queues_);

      gaspi_queue_id_t get() const;
      gaspi_queue_id_t swap(gaspi_queue_id_t badQueue);

    private:
      inline gaspi_queue_id_t stateToQueue(const long) const;
      inline void clearQueue(const gaspi_queue_id_t queue);

      static const long CACHE_LINE_SIZE = 64;
      const long numQueues;

      char pad0 [CACHE_LINE_SIZE];
      volatile long state;
      char pad1 [CACHE_LINE_SIZE];

      std::vector<gaspi_queue_id_t> queueStock;
    };
  }
}
