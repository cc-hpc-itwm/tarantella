#pragma once

#include "gpi/Types.hpp"

#include <GASPI.h>

#include <random>
#include <vector>

namespace tarantella
{
  namespace GPI
  {
    class QueueManager
    {
      public:
        static QueueManager& get_instance();
        QueueManager(QueueManager const&) = delete;
        QueueManager& operator=(QueueManager const&) = delete;
        ~QueueManager();

        QueueID get_queue_id_for_write_notify();
        void wait_and_flush_queue();

      private:
        QueueManager();
        QueueID wait_and_flush_queue(QueueID);

        // Assumption: the IDs of the preallocated queues are in the 
        // [0, num_preallocated_queues-1) range
        std::size_t const num_preallocated_queues;
        std::vector<QueueID> gaspi_queues;
        std::size_t const slots_per_gaspi_queue;
        std::mt19937 rng;
    };
  }
}