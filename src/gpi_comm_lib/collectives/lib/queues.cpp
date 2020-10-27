#include "queues.h"
#include "gpi/gaspiCheckReturn.hpp"

namespace tarantella
{
  namespace collectives
  {
    using tarantella::GPI::gaspiCheckReturn;

    queues::queues(const long num,
                  const gaspi_queue_id_t first)
    : numQueues(num)
    , state(0) {
      for (long i=first; i < first + num; i++) {
        queueStock.push_back(i);
      }
    }

    queues::queues(const std::vector<gaspi_queue_id_t>& queues_)
      : numQueues(queues_.size()),
        state(0),
        queueStock(queues_) {
    }

    gaspi_queue_id_t queues::get() const {
      return stateToQueue(state);
    }

    inline gaspi_queue_id_t queues::stateToQueue(const long state_) const {
      return queueStock[state_ % numQueues];
    }

    gaspi_queue_id_t queues::swap(gaspi_queue_id_t badQueue) {
      const long stateLocal = state;
      const gaspi_queue_id_t queueLocal = stateToQueue(stateLocal);

      if (queueLocal != badQueue) {
        return queueLocal;
      } else {
        const long stateLocalNew = stateLocal + 1;
        const gaspi_queue_id_t queueLocalNew = stateToQueue(stateLocalNew);

        clearQueue(queueLocalNew);

        const long stateBeforeSwap =
          __sync_val_compare_and_swap(&state, stateLocal, stateLocalNew);

        return (stateBeforeSwap == stateLocal)
              ? queueLocalNew
              : stateToQueue(stateBeforeSwap);
      };
    }

    inline void queues::clearQueue(const gaspi_queue_id_t queue) {
      gaspiCheckReturn(gaspi_wait(queue, GASPI_BLOCK),
                      "Failed to clear queue with ");
    }
  }
}
