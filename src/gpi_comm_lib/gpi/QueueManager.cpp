#include "QueueManager.hpp"

#include "gpi/gaspiCheckReturn.hpp"

#include <GASPI.h>

#include <algorithm>
#include <functional>
#include <numeric>
#include <random>

namespace tarantella
{
  namespace GPI
  {
    namespace
    {
      std::size_t get_slots_per_gaspi_queue()
      {
        gaspi_number_t slots;
        gaspiCheckReturn(gaspi_queue_size_max(&slots),
                        "[QueueManager::get_slots_per_gaspi_queue()] GASPI:\
                         Error in gaspi_queue_size_max");
        return slots;
      }

      std::size_t get_number_allocated_gaspi_queues()
      {
        gaspi_number_t number_queues;
        gaspiCheckReturn(gaspi_queue_num(&number_queues),
                         "[QueueManager::get_number_allocated_gaspi_queues()] GASPI:\
                          Could not get number of allocated queues");
        return static_cast<std::size_t>(number_queues);
      }

      std::size_t get_number_gaspi_queues()
      {
        std::size_t const number_queues_want_to_use = 10;
        gaspi_number_t number_queues_allowed;
        gaspiCheckReturn(gaspi_queue_max(&number_queues_allowed),
                         "[QueueManager::get_number_gaspi_queues()] GASPI:\
                          Could not get max number of queues");
        return std::min(number_queues_want_to_use,
                        static_cast<std::size_t>(number_queues_allowed));
      }

      auto queue_has_two_empty_slots(std::size_t total_slots)
      {
        return [total_slots](auto queue)
        {
          gaspi_number_t non_empty_slots;
          gaspiCheckReturn(gaspi_queue_size(queue, &non_empty_slots),
                          "[QueueManager::queue_has_two_empty_slots()] GASPI:\
                           Error in gaspi_queue_size");
          return total_slots >= non_empty_slots + 2;
        };
      }
    }

    QueueManager& QueueManager::get_instance()
    {
      static auto instance = new QueueManager();
      return *instance;
    }

    QueueManager::QueueManager()
    : num_preallocated_queues(get_number_allocated_gaspi_queues()),
      gaspi_queues(get_number_gaspi_queues()),
      slots_per_gaspi_queue(get_slots_per_gaspi_queue()),
      rng(std::random_device()())
    {
      auto const end = std::min(gaspi_queues.size(), num_preallocated_queues);
      std::iota(gaspi_queues.begin(), gaspi_queues.begin() + end, 0);

      // allocate remaining queues
      if (num_preallocated_queues < gaspi_queues.size())
      {
        auto const start_unallocated_queues_it = gaspi_queues.begin() + num_preallocated_queues;
        auto const num_unallocated_queues = gaspi_queues.size() - num_preallocated_queues;
        std::generate_n(start_unallocated_queues_it, num_unallocated_queues,
                        []() {
                          gaspi_queue_id_t q;
                          gaspiCheckReturn(gaspi_queue_create(&q, GASPI_BLOCK),
                                          "[QueueManager::QueueManager()] GASPI:\
                                            Could not create queue");
                          return q;
                        });
      }
    }

    QueueManager::~QueueManager()
    {
      wait_and_flush_queue();

      // only delete the queues allocated by the manager
      std::sort(gaspi_queues.begin(), gaspi_queues.end(), std::greater<QueueID>());
      if (num_preallocated_queues < gaspi_queues.size())
      {
        for (auto q = gaspi_queues.begin(); q != gaspi_queues.end() - num_preallocated_queues; ++q)
        {
          gaspiCheckReturn(gaspi_queue_delete(*q),
                           "[QueueManager::QueueManager()] GASPI: Could not delete queue");
        }
      }
    }

    QueueID QueueManager::get_queue_id_for_write_notify()
    {
      std::shuffle(gaspi_queues.begin(), gaspi_queues.end(), rng);
      auto const valid_queue = std::find_if(gaspi_queues.begin(), gaspi_queues.end(),
                                            queue_has_two_empty_slots(slots_per_gaspi_queue));
      if(valid_queue != gaspi_queues.end()) return *valid_queue;
      else return wait_and_flush_queue(gaspi_queues.front());
    }

    void QueueManager::wait_and_flush_queue()
    {
      for(auto q : gaspi_queues) wait_and_flush_queue(q);
    }

    QueueID QueueManager::wait_and_flush_queue(QueueID id)
    {
      gaspiCheckReturn(gaspi_wait(id, GASPI_BLOCK),
                       "[QueueManager::wait_and_flush_queue()] GASPI:\
                        Error while waiting on queue");
      return id;
    }
  }
}