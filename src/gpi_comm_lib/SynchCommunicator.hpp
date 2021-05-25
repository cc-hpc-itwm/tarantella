#pragma once

#include "FusedTensorInfo.hpp"
#include "TensorInfo.hpp"

#include <GaspiCxx/collectives/non_blocking/Collective.hpp>
#include <GaspiCxx/collectives/non_blocking/collectives_lowlevel/AllreduceRing.hpp>
#include <GaspiCxx/collectives/non_blocking/Allreduce.hpp>
#include <GaspiCxx/group/Group.hpp>

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
  using Collective = gaspi::collectives::Collective;

  class SynchCommunicator
  {
    public:
      SynchCommunicator(gaspi::group::Group const&, std::vector<collectives::TensorInfo> const&);
      SynchCommunicator(gaspi::group::Group const&, std::vector<collectives::TensorInfo> const&, std::size_t);
      SynchCommunicator(SynchCommunicator&) = delete;
      SynchCommunicator& operator=(SynchCommunicator&) = delete;
      ~SynchCommunicator() = default;

      // TODO: Replace void* with a LocalBuffer struct {ptr, size}
      void start_allreduce_impl(GradID const&, const void*);
      void finish_allreduce_impl(GradID const&, void*);

    private:
      static gaspi::collectives::ReductionOp const reduction_op;

      std::unordered_map<GradID, FusedID> fused_ids;
      std::unordered_map<FusedID, collectives::FusedTensorInfo> fused_tensor_infos;
      std::unordered_map<FusedID, std::unique_ptr<Collective>> operators;
      std::unordered_map<FusedID, std::vector<char>> fused_buffers;

      std::unordered_map<FusedID, std::unique_ptr<std::atomic<std::size_t>>> ready_to_start_counters;
      std::unordered_map<FusedID, std::unique_ptr<std::atomic<std::size_t>>> finished_counters;
      std::unordered_map<FusedID, std::unique_ptr<std::atomic<bool>>> ready_to_copy_back;
      std::unordered_map<FusedID, std::unique_ptr<std::atomic<std::size_t>>> ready_to_reset_counters;

      void create_fused_tensor_infos_and_ids(std::vector<collectives::TensorInfo> const&, std::size_t);
      void create_fused_tensors_synchronization();

      void create_fused_tensor_infos(std::vector<collectives::TensorInfo> const &tensor_infos);

      template <typename T, gaspi::collectives::AllreduceAlgorithm>
      void create_operators(gaspi::group::Group const&);

      void create_fused_buffers();
      void copy_data_to_fused_buffer(GradID const&, const void*);
      void copy_data_from_fused_buffer(GradID const&, void*);

  };

  template <typename T, gaspi::collectives::AllreduceAlgorithm Algorithm>
  void SynchCommunicator::create_operators(gaspi::group::Group const& group)
  {
    for(auto const& fused_info : fused_tensor_infos)
    {
      auto const tensor_id = fused_info.first;
      auto const tensor_info = fused_info.second.to_tensor_info();
      operators.emplace(tensor_id,
                        std::make_unique<gaspi::collectives::Allreduce<T, Algorithm>>(
                          group, tensor_info.get_nelems(), reduction_op));
    }
  }
}
