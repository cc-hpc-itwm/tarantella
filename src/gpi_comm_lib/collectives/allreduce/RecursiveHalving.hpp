#pragma once

#include "Operator.hpp"
#include "allreduceButterfly.h"
#include "collectives/barrier/GPIBarrier.hpp"
#include "collectives/TensorInfo.hpp"
#include "gpi/Group.hpp"
#include "gpi/NotificationManager.hpp"
#include "gpi/SegmentBuffer.hpp"

namespace tarantella
{
  namespace collectives
  {
    namespace Allreduce
    {
      class RecursiveHalving : public Operator
      {
        public:
          RecursiveHalving(TensorInfo,
                           ReductionOp,
                           ResourceList const&,
                           queues&,
                           GPI::Group const&);
          RecursiveHalving(const RecursiveHalving&) = delete;
          RecursiveHalving& operator=(const RecursiveHalving&) = delete;
          ~RecursiveHalving() = default;
  
          void start() override;
          void trigger_communication_step() override;
  
          void reset_for_reuse() override;
          bool is_running() const override;
          bool is_finished() const override;
  
          void* get_input_ptr() const override;
          void* get_result_ptr() const override;
  
          static RequiredResourceList get_required_resources(TensorInfo const&, GPI::Group const&);
  
        private:
          GPI::Group const& group;
          std::atomic<OperatorState> state;
          allreduceButterfly allreduce;
          Barrier::GPIBarrier barrier;
      };
    }
  }
}