#include "SynchCommunicator.hpp"

#include <cstring>
#include <memory>
#include <utility>

namespace tarantella
{
  gaspi::collectives::ReductionOp const SynchCommunicator::reduction_op =
                                        gaspi::collectives::ReductionOp::SUM;

  void SynchCommunicator::create_fused_tensor_infos_and_ids(
    std::vector<collectives::TensorInfo> const& tensor_infos,
    std::size_t threshold_bytes)
  {
    collectives::TensorFusor fusor {threshold_bytes};
    fusor.fuse_tensor_infos_and_ids(tensor_infos, fused_ids, fused_tensor_infos);
  }

  void SynchCommunicator::create_fused_tensors_synchronization()
  {
    for(auto const& fused_info : fused_tensor_infos)
    {
      auto const fused_id = fused_info.first;
      ready_to_start_counters[fused_id] = std::make_unique<std::atomic<std::size_t>>(0UL);
      finished_counters[fused_id] = std::make_unique<std::atomic<std::size_t>>(0UL);
      ready_to_copy_back[fused_id] = std::make_unique<std::atomic<bool>>(false);
      ready_to_reset_counters[fused_id] = std::make_unique<std::atomic<std::size_t>>(0UL);
    }
  }

  SynchCommunicator::SynchCommunicator(gaspi::group::Group const& group,
                                       std::vector<collectives::TensorInfo> const& tensor_infos,
                                       std::size_t threshold_for_tensor_fusion_bytes)
  : fused_ids(),
    fused_tensor_infos(),
    operators(),
    fused_buffers(),
    ready_to_start_counters(),
    finished_counters(),
    ready_to_copy_back(),
    ready_to_reset_counters()
  {
    using T = float;
    auto const algorithm = gaspi::collectives::AllreduceAlgorithm::RING;

    create_fused_tensor_infos_and_ids(tensor_infos, threshold_for_tensor_fusion_bytes);
    create_fused_tensors_synchronization();
    create_operators<T, algorithm>(group);
    create_fused_buffers();
  }

  SynchCommunicator::SynchCommunicator(gaspi::group::Group const& group,
                                       std::vector<collectives::TensorInfo> const& tensor_infos)
  : SynchCommunicator(group, tensor_infos, 0UL)
  { }

  void SynchCommunicator::start_allreduce_impl(GradID const& grad_id, const void* data_ptr)
  {
    auto const fused_id = fused_ids[grad_id];

    const void* source_buffer = fused_buffers[fused_id].data();
    if (fused_tensor_infos[fused_id].get_num_tensors() > 1)
    {
      copy_data_to_fused_buffer(grad_id, data_ptr);
    }
    else
    {
      source_buffer = data_ptr;
    }
    
    auto const value = ready_to_start_counters[fused_id]->fetch_add(1UL);

    // Make sure all copies are done, before last `grad_id` starts operator
    if (value == fused_tensor_infos[fused_id].get_num_tensors()-1)
    {
      operators[fused_id]->start(source_buffer);
      ready_to_start_counters[fused_id]->store(0UL);
    }
  }

  void SynchCommunicator::finish_allreduce_impl(GradID const& grad_id, void* results_ptr)
  {
    auto const fused_id = fused_ids[grad_id];

    void* destination_buffer = fused_buffers[fused_id].data();
    if (fused_tensor_infos[fused_id].get_num_tensors() == 1)
    {
      destination_buffer = results_ptr;
    }

    // First `grad_id` to arrive waits for `has_finished`, and notifies
    // everyone that results can be copied back
    auto const num_arrived = finished_counters[fused_id]->fetch_add(1UL);
    if (num_arrived == 0)
    {
      operators[fused_id]->waitForCompletion(destination_buffer);
      ready_to_copy_back[fused_id]->store(true);
    }

    // All `grad_id`s copy-out their respective data,
    // once results have been obtained
    while(true)
    {
      if(ready_to_copy_back[fused_id]->load())
      {
        if (fused_tensor_infos[fused_id].get_num_tensors() > 1)
        {
           copy_data_from_fused_buffer(grad_id, results_ptr);
        }
        break;
      }
    }

    // Make sure all copies are done, before last `grad_id` resets initial state
    auto const copied_grads = ready_to_reset_counters[fused_id]->fetch_add(1UL);
    if (copied_grads == fused_tensor_infos[fused_id].get_num_tensors()-1)
    {
      finished_counters[fused_id]->store(0UL);
      ready_to_copy_back[fused_id]->store(false);
      ready_to_reset_counters[fused_id]->store(0UL);
    }
  }

  void SynchCommunicator::copy_data_to_fused_buffer(GradID const& grad_id, const void* data_ptr)
  {
    auto const fused_id = fused_ids[grad_id];
    auto const fused_buffer_ptr = fused_buffers[fused_id].data() +
                                  fused_tensor_infos[fused_id].get_local_offset_bytes(grad_id);
    std::memcpy(fused_buffer_ptr, data_ptr,
                fused_tensor_infos[fused_id].get_local_size_bytes(grad_id));
  }

  void SynchCommunicator::copy_data_from_fused_buffer(GradID const& grad_id, void* results_ptr)
  {
    auto const fused_id = fused_ids[grad_id];
    auto const fused_buffer_ptr = fused_buffers[fused_id].data() +
                                  fused_tensor_infos[fused_id].get_local_offset_bytes(grad_id);
    std::memcpy(results_ptr, fused_buffer_ptr,
                fused_tensor_infos[fused_id].get_local_size_bytes(grad_id));
  }

  void SynchCommunicator::create_fused_buffers()
  {
    for(auto const& fused_info : fused_tensor_infos)
    {
      auto const tensor_id = fused_info.first;
      auto const tensor_info = fused_info.second.to_tensor_info();
      fused_buffers.emplace(tensor_id, std::vector<char>(tensor_info.get_size_bytes()));
    }
  }
}

