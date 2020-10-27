#pragma once

#include "Operator.hpp"
#include "allreduceButterflyDoubleBuffer.h"
#include "collectives/TensorInfo.hpp"
#include "gpi/Group.hpp"

namespace tarantella
{
  namespace collectives
  {
    namespace Allreduce
    {
      class RecursiveHalvingDoubleBuffer : public Operator
      {
        public:
          RecursiveHalvingDoubleBuffer(TensorInfo,
                                      ReductionOp,
                                      ResourceList const&,
                                      queues&,
                                      GPI::Group const&);
          RecursiveHalvingDoubleBuffer(const RecursiveHalvingDoubleBuffer&) = delete;
          RecursiveHalvingDoubleBuffer& operator=(const RecursiveHalvingDoubleBuffer&) = delete;
          ~RecursiveHalvingDoubleBuffer() = default;

          void start() override;
          void trigger_communication_step() override;

          void reset_for_reuse() override;
          bool is_running() const override;
          bool is_finished() const override;

          virtual void* get_input_ptr() const override;
          virtual void* get_result_ptr() const override;

          static RequiredResourceList get_required_resources(TensorInfo const&, GPI::Group const& group);

        private:
          std::atomic<OperatorState> state;
          allreduceButterflyDoubleBuffer allreduce;
      };
    }
  }
}
