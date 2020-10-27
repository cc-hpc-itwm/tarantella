#include "SynchCommunicator.hpp"
#include "collectives/allreduce/RecursiveHalvingDoubleBuffer.hpp"

#include <cstring>
#include <memory>
#include <utility>

namespace tarantella
{
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

  SynchCommunicator::SynchCommunicator(GPI::Context& context,
                                       GPI::SegmentID segment_id,
                                       GPI::Group const& group,
                                       std::vector<collectives::TensorInfo> const& tensor_infos,
                                       std::size_t threshold_for_tensor_fusion_bytes)
  : resource_manager(context.get_resource_manager()),
    segment_id(segment_id),
    group(group),
    queue_handler(),
    fused_ids(),
    fused_tensor_infos(),
    operators(),
    ready_to_start_counters(),
    finished_counters(),
    ready_to_copy_back(),
    ready_to_reset_counters(),
    setup_has_finished(),
    terminate_man_thread(false),
    management_thread(&tarantella::SynchCommunicator::management_thread_task, this)
  {
    using AllreduceImplementation = collectives::Allreduce::RecursiveHalvingDoubleBuffer;
    create_fused_tensor_infos_and_ids(tensor_infos, threshold_for_tensor_fusion_bytes);
    create_fused_tensors_synchronization();
    create_segment_resources<AllreduceImplementation>(tensor_infos);
    create_operators_with_state<AllreduceImplementation>();
    setup_has_finished.notify();
  }

  SynchCommunicator::SynchCommunicator(GPI::Context& context,
                                       GPI::SegmentID segment_id,
                                       GPI::Group const& group,
                                       std::vector<collectives::TensorInfo> const& tensor_infos)
  : SynchCommunicator(context, segment_id, group, tensor_infos, 0UL)
  { }

  SynchCommunicator::~SynchCommunicator()
  {
    terminate_man_thread = true;
    if (management_thread.joinable())
    {
      management_thread.join();
    }
  }

  void SynchCommunicator::start_allreduce_impl(GradID const& grad_id, const void* data_ptr)
  {
    auto const fused_id = fused_ids[grad_id];

    // All `grad_id`s copy-in their respective data
    copy_data_to_segment(grad_id, data_ptr);
    auto const value = ready_to_start_counters[fused_id]->fetch_add(1UL);

    // Make sure all copies are done, before last `grad_id` starts operator
    if (value == fused_tensor_infos[fused_id].get_num_tensors()-1)
    {
      operators[fused_id].allreduce->start();
      ready_to_start_counters[fused_id]->store(0UL);
    }
  }

  void SynchCommunicator::finish_allreduce_impl(GradID const& grad_id, void* results_ptr)
  {
    auto const fused_id = fused_ids[grad_id];

    // First `grad_id` to arrive waits for `has_finished`, and notifies
    // everyone that results can be copied back
    auto const num_arrived = finished_counters[fused_id]->fetch_add(1UL);
    if (num_arrived == 0)
    {
      operators[fused_id].has_finished->wait();
      ready_to_copy_back[fused_id]->store(true);
    }

    // All `grad_id`s copy-out their respective data,
    // once results have been obtained
    while(true)
    {
      if(ready_to_copy_back[fused_id]->load())
      {
        copy_data_from_segment(grad_id, results_ptr);
        break;
      }
    }

    // Make sure all copies are done, before last `grad_id` resets initial state
    auto const copied_grads = ready_to_reset_counters[fused_id]->fetch_add(1UL);
    if (copied_grads == fused_tensor_infos[fused_id].get_num_tensors()-1)
    {
      operators[fused_id].allreduce->reset_for_reuse();
      finished_counters[fused_id]->store(0UL);
      ready_to_copy_back[fused_id]->store(false);
      ready_to_reset_counters[fused_id]->store(0UL);
    }
  }

  void SynchCommunicator::copy_data_to_segment(GradID const& grad_id, const void* data_ptr)
  {
    auto const fused_id = fused_ids[grad_id];
    auto const segment_ptr = reinterpret_cast<char*>(operators[fused_id].allreduce->get_input_ptr())
                             + fused_tensor_infos[fused_id].get_local_offset_bytes(grad_id);
    std::memcpy(segment_ptr, data_ptr, fused_tensor_infos[fused_id].get_local_size_bytes(grad_id));
  }

  void SynchCommunicator::copy_data_from_segment(GradID const& grad_id, void* results_ptr)
  {
    auto const fused_id = fused_ids[grad_id];
    auto const segment_ptr = reinterpret_cast<char*>( operators[fused_id].allreduce->get_result_ptr())
                             + fused_tensor_infos[fused_id].get_local_offset_bytes(grad_id);
    std::memcpy(results_ptr, segment_ptr, fused_tensor_infos[fused_id].get_local_size_bytes(grad_id));
  }

  void SynchCommunicator::management_thread_task()
  {
    setup_has_finished.wait();
    while (!terminate_man_thread)
    {
      while (true)
      {
        if (terminate_man_thread)
        {
          break;
        }
        for (auto& element : operators)
        {
          auto& op = *(element.second.allreduce.get());
          if (op.is_finished()) continue;

          op.trigger_communication_step();
          if (op.is_finished())
          {
            element.second.has_finished->notify();
          }
        }
      }
    }
  }
}
