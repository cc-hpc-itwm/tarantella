#pragma once

#include "collectives/allreduce/Operator.hpp"
#include "collectives/barrier/GPIBarrier.hpp"
#include "collectives/allreduce/Types.hpp"
#include "collectives/TensorInfo.hpp"
#include "gpi/Context.hpp"
#include "gpi/Group.hpp"
#include "queues.h"

#include <memory>
#include <vector>

namespace tarantella
{
  class TensorAllreducer
  {
    public:
      TensorAllreducer(GPI::Context&,
                       GPI::SegmentID,
                       GPI::Group const&,
                       collectives::Allreduce::ReductionOp const&,
                       std::vector<collectives::TensorInfo> const&
                      );
      void exec_allreduce(std::vector<const void*> data_ptrs,
                          std::vector<void*> output_ptrs);

    private:
      GPI::Context& context;
      GPI::Group const group;
      collectives::Allreduce::ReductionOp reduction_op;
      collectives::queues queue_handler;
      collectives::Barrier::GPIBarrier barrier;
      GPI::ResourceManager& resource_manager;

      std::unique_ptr<collectives::Allreduce::Operator> allreduce_op;
      std::vector<std::size_t> tensor_sizes_bytes;

      collectives::TensorInfo create_unified_tensor(std::vector<collectives::TensorInfo> const&);
      template <typename AllreduceAlgorithm>
      void create_resources_and_allreduce_op(collectives::TensorInfo const&,
                                             GPI::SegmentID const&);
      void copy_data_to_segment(const void*, std::size_t);
      void copy_data_from_segment(void*, std::size_t);
  };
}