#include "RecursiveHalving.hpp"

#include "utils.hpp"

namespace tarantella
{
  namespace collectives
  {
    namespace Allreduce
    {
      RecursiveHalving::RecursiveHalving(TensorInfo tensor_info,
                                         ReductionOp reduction_op,
                                         ResourceList const &resource_list,
                                         queues &queues,
                                         GPI::Group const &group)
          : group(group),
            state(OperatorState::NOT_STARTED),
            allreduce(tensor_info.get_nelems(), to_allreduce_dataType(tensor_info.get_elem_type()),
                      to_allreduce_reductionType(reduction_op),
                      to_allreduce_segment_buffer(resource_list.at(0)),
                      to_allreduce_segment_buffer(resource_list.at(1)),
                      queues, group),
            barrier(group)
      {}
  
      void RecursiveHalving::start()
      {
        if (is_running())
        {
          throw std::logic_error("[RecursiveHalving::start] Operation already started.");
        }
        if (is_finished())
        {
          throw std::logic_error("[RecursiveHalving::start] Operation not reset after finish.");
        }
        allreduce.signal();
        state = OperatorState::RUNNING;
      }
  
      void RecursiveHalving::trigger_communication_step()
      {
        if (is_running())
        {
          auto const result = allreduce();
          if (result == 0)
          {
            barrier.blocking_barrier();
            state = OperatorState::FINISHED;
          }
        }
        else
        {
          // do nothing before start() is called
        }
      }
  
      void RecursiveHalving::reset_for_reuse()
      {
        if (is_running())
        {
          throw std::logic_error("[RecursiveHalving::reset] Cannot reset while running.");
        }
        state = OperatorState::NOT_STARTED;
      }
  
      bool RecursiveHalving::is_running() const
      {
        return state == OperatorState::RUNNING;
      }
  
      bool RecursiveHalving::is_finished() const
      {
        return state == OperatorState::FINISHED;
      }
  
      Operator::RequiredResourceList RecursiveHalving::get_required_resources(
        TensorInfo const& tensor_info, GPI::Group const& group)
      {
        auto const num_notifications = allreduceButterfly::getNumberOfNotifications(group.get_size());
        auto const num_elements_data_segment = tensor_info.get_nelems();
        auto const num_elements_temp_segment = static_cast<size_t>(
            allreduceButterfly::getNumberOfElementsSegmentCommunicate(tensor_info.get_nelems(), group.get_size()));
        return {{num_elements_data_segment * getDataTypeSize(tensor_info.get_elem_type()), num_notifications},
                {num_elements_temp_segment * getDataTypeSize(tensor_info.get_elem_type()), num_notifications}};
      }
  
      void* RecursiveHalving::get_input_ptr() const
      {
        return allreduce.getReducePointer();
      }
  
      void* RecursiveHalving::get_result_ptr() const
      {
        return allreduce.getReducePointer();
      }
    }
  }
}
