#pragma once

#include "collectives/BufferElementType.hpp"
#include "gpi/NotificationManager.hpp"
#include "gpi/SegmentBuffer.hpp"

#include <cstddef>
#include <vector>

namespace tarantella
{
  namespace collectives
  {
    namespace Allreduce
    {
      // \note
      // Interface for non-blocking, asynchronous Allreduce algorithms (not thread-safe)
      class Operator
      {
        public:
          class RequiredResource
          {
            public:
              std::size_t buffer_size;
              std::size_t num_notifications;
          };
          using RequiredResourceList = std::vector<RequiredResource>;
          using Resource = std::pair<GPI::SegmentBuffer, GPI::NotificationManager::NotificationRange>;
          using ResourceList = std::vector<Resource>;
  
          enum class ReductionOp
          {
            SUM,
            AVERAGE
          };
  
          enum class OperatorState
          {
            NOT_STARTED,
            RUNNING,
            FINISHED
          };
  
          virtual ~Operator() = default;
  
          // Initiates the Allreduce operation (non-blocking)
          // and sets is_running == TRUE
          virtual void start() = 0;
  
          // Makes partial progress towards computing the Allreduce result
          // and has to be called multiple times until the operation is completed,
          // when is_finished == TRUE
          // can be called independently of the state;
          // it only tries to make progress if is_running == TRUE
          virtual void trigger_communication_step() = 0;
  
          // Enables the Allreduce to be started again
          // and sets is_running == FALSE and is_finished == FALSE
          virtual void reset_for_reuse() = 0;
          virtual bool is_running() const = 0;
  
          // If TRUE, results are available until reset_for_reuse() is called
          virtual bool is_finished() const = 0;
  
          // TODO: void* -> SegmentBuffer
          virtual void* get_input_ptr() const = 0;
          virtual void* get_result_ptr() const = 0;
      };
    }
  }
}
