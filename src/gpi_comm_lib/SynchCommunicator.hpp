#pragma once

#include "AtomicCondition.hpp"
#include "collectives/allreduce/Operator.hpp"
#include "collectives/barrier/GPIBarrier.hpp"
#include "collectives/FusedTensorInfo.hpp"
#include "collectives/TensorInfo.hpp"
#include "collectives/Types.hpp"
#include "distribution/utilities.hpp"
#include "gpi/Context.hpp"
#include "gpi/ResourceManager.hpp"
#include "queues.h"

#include <atomic>
#include <cstddef>
#include <memory>
#include <thread>
#include <typeindex>
#include <unordered_map>
#include <vector>

namespace tarantella
{
  using GradID = collectives::GradID;
  using FusedID = collectives::FusedID;

  class SynchCommunicator
  {
    public:
      SynchCommunicator(GPI::Context&, GPI::SegmentID, GPI::Group const&, std::vector<collectives::TensorInfo> const&);
      SynchCommunicator(GPI::Context&, GPI::SegmentID, GPI::Group const&, std::vector<collectives::TensorInfo> const&, std::size_t);
      SynchCommunicator(SynchCommunicator&) = delete;
      SynchCommunicator& operator=(SynchCommunicator&) = delete;
      ~SynchCommunicator();

      // TODO: Replace void* with a LocalBuffer struct {ptr, size}
      void start_allreduce_impl(GradID const&, const void*);
      void finish_allreduce_impl(GradID const&, void*);

    private:
      struct OperatorWithState
      {
        std::unique_ptr<collectives::Allreduce::Operator> allreduce;
        std::unique_ptr<AtomicCondition> has_finished;
      };

      static collectives::Allreduce::Operator::ReductionOp const reduction_op = collectives::Allreduce::Operator::ReductionOp::AVERAGE;

      GPI::ResourceManager& resource_manager;
      GPI::SegmentID segment_id;
      GPI::Group const& group;
      collectives::queues queue_handler; // TODO replace with the ResourceManager

      std::unordered_map<GradID, FusedID> fused_ids;
      std::unordered_map<FusedID, collectives::FusedTensorInfo> fused_tensor_infos;
      std::unordered_map<FusedID, OperatorWithState> operators;

      std::unordered_map<FusedID, std::unique_ptr<std::atomic<std::size_t>>> ready_to_start_counters;
      std::unordered_map<FusedID, std::unique_ptr<std::atomic<std::size_t>>> finished_counters;
      std::unordered_map<FusedID, std::unique_ptr<std::atomic<bool>>> ready_to_copy_back;
      std::unordered_map<FusedID, std::unique_ptr<std::atomic<std::size_t>>> ready_to_reset_counters;

      AtomicCondition setup_has_finished;
      std::atomic<bool> terminate_man_thread;
      std::thread management_thread;
      void management_thread_task();

      void copy_data_to_segment(GradID const&, const void*);
      void copy_data_from_segment(GradID const&, void*);
      
      void create_fused_tensor_infos_and_ids(std::vector<collectives::TensorInfo> const&, std::size_t);
      void create_fused_tensors_synchronization();

      template <typename AllreduceAlgorithm>
      constexpr float get_overhead_factor() const;

      template <typename AllreduceAlgorithm>
      void create_segment_resources(std::vector<collectives::TensorInfo> const& tensor_infos) const;

      void create_fused_tensor_infos(std::vector<collectives::TensorInfo> const &tensor_infos);

      template <typename AllreduceAlgorithm>
      std::unique_ptr<collectives::Allreduce::Operator> create_allreduce_op(collectives::TensorInfo const&);

      template <typename AllreduceAlgorithm>
      void create_operators_with_state();
  };

  template <typename AllreduceAlgorithm>
  constexpr float SynchCommunicator::get_overhead_factor() const
  {
    return 3.5;
  }

  template <typename AllreduceAlgorithm>
  void SynchCommunicator::create_segment_resources(std::vector<collectives::TensorInfo> const& tensor_infos) const
  {
    auto const segment_size = distribution::get_segment_size(tensor_infos, get_overhead_factor<AllreduceAlgorithm>());
    resource_manager.make_segment_resources(segment_id, group, segment_size);

    // Barrier is required, to ensure all ranks have finished registering
    // their segments to their communication partners
    collectives::Barrier::GPIBarrier barrier(group);
    barrier.blocking_barrier();
  }

  template <typename AllreduceAlgorithm>
  std::unique_ptr<collectives::Allreduce::Operator> SynchCommunicator::create_allreduce_op(collectives::TensorInfo const& tensor_info)
  {
    auto const required_resources = AllreduceAlgorithm::get_required_resources(tensor_info, group);

    collectives::Allreduce::Operator::ResourceList resources;
    for (auto const& resource : required_resources)
    {
      resources.emplace_back(
          resource_manager.get_buffer_of_size(segment_id, resource.buffer_size),
          resource_manager.get_notification_range(segment_id, resource.num_notifications));
    }

    return std::make_unique<AllreduceAlgorithm>(tensor_info, reduction_op, resources, queue_handler, group);
  }

  template <typename AllreduceAlgorithm>
  void SynchCommunicator::create_operators_with_state()
  {
    for(auto const& fused_info : fused_tensor_infos)
    {
      auto const tensor_id = fused_info.first;
      auto const tensor_info = fused_info.second.to_tensor_info();
      OperatorWithState op{create_allreduce_op<AllreduceAlgorithm>(tensor_info), std::make_unique<AtomicCondition>()};
      operators.emplace(tensor_id, std::move(op));
    }
  }
}
