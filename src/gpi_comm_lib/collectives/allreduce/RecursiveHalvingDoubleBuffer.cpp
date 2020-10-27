#include "RecursiveHalvingDoubleBuffer.hpp"

#include "gpi/gaspiCheckReturn.hpp"
#include "utils.hpp"

namespace tarantella
{
  namespace collectives
  {
    namespace Allreduce
    {
      RecursiveHalvingDoubleBuffer::RecursiveHalvingDoubleBuffer(TensorInfo tensor_info,
                                                                 ReductionOp reduction_op,
                                                                 ResourceList const& resource_list,
                                                                 queues& queues,
                                                                 GPI::Group const& group) 
      : state(OperatorState::NOT_STARTED),
        allreduce(tensor_info.get_nelems(),
                  to_allreduce_dataType(tensor_info.get_elem_type()),
                  to_allreduce_reductionType(reduction_op),
                  to_allreduce_segment_buffer(resource_list.at(0)),
                  to_allreduce_segment_buffer(resource_list.at(1)),
                  to_allreduce_segment_buffer(resource_list.at(2)),
                  queues, group)
      { }

      void RecursiveHalvingDoubleBuffer::start()
      {
        if (is_running())
        {
          throw std::logic_error("[RecursiveHalvingDoubleBuffer::start] Operation already started.");
        }
        if (is_finished())
        {
          throw std::logic_error("[RecursiveHalvingDoubleBuffer::start] Operation not reset after finish.");
        }
        allreduce.signal();
        state = OperatorState::RUNNING;
      }

      void RecursiveHalvingDoubleBuffer::trigger_communication_step()
      {
        if (is_running())
        {
          auto const result = allreduce();
          if (result == 0)
          {
            state = OperatorState::FINISHED;
          }
        }
      }

      void RecursiveHalvingDoubleBuffer::reset_for_reuse()
      {
        if (is_running())
        {
          throw std::logic_error("[RecursiveHalvingDoubleBuffer::reset] Cannot reset while running.");
        }
        state = OperatorState::NOT_STARTED;
      }

      bool RecursiveHalvingDoubleBuffer::is_running() const
      {
        return state == OperatorState::RUNNING;
      }

      bool RecursiveHalvingDoubleBuffer::is_finished() const
      {
        return state == OperatorState::FINISHED;
      }

      Operator::RequiredResourceList RecursiveHalvingDoubleBuffer::get_required_resources(
          TensorInfo const& tensor_info, GPI::Group const& group)
      {
        auto const num_notifications = allreduceButterflyDoubleBuffer::getNumberOfNotifications(group.get_size());

        auto const num_elements_data_segment = tensor_info.get_nelems();
        auto const num_elements_temp_segment = static_cast<size_t>(
            allreduceButterflyDoubleBuffer::getNumberOfElementsSegmentCommunicate(
                tensor_info.get_nelems(), group.get_size()));

        return {{num_elements_data_segment * getDataTypeSize(tensor_info.get_elem_type()), num_notifications},
                {num_elements_data_segment * getDataTypeSize(tensor_info.get_elem_type()), num_notifications},
                {num_elements_temp_segment * getDataTypeSize(tensor_info.get_elem_type()), num_notifications}};
      }

      void* RecursiveHalvingDoubleBuffer::get_input_ptr() const
      {
        return allreduce.getActiveReducePointer();
      }

      void* RecursiveHalvingDoubleBuffer::get_result_ptr() const
      {
        return allreduce.getResultsPointer();
      }
    }
  }
}
