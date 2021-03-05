#include "TensorAllreducer.hpp"
#include "gpi/ResourceManager.hpp"
#include "collectives/allreduce/RecursiveHalvingDoubleBuffer.hpp"

#include <cstring>

namespace tarantella
{
  TensorAllreducer::TensorAllreducer(GPI::Context& context,
                                     GPI::SegmentID segment_id,
                                     GPI::Group const& group,
                                     collectives::Allreduce::ReductionOp const& reduction_op,
                                     std::vector<collectives::TensorInfo> const& tensor_infos)
  : context(context),
    group(group),
    reduction_op(reduction_op),
    queue_handler(),
    barrier(group),
    resource_manager(context.get_resource_manager())
  {
    using AllreduceAlgorithm = collectives::Allreduce::RecursiveHalvingDoubleBuffer;
    create_resources_and_allreduce_op<AllreduceAlgorithm>(create_unified_tensor(tensor_infos),
                                                          segment_id);
  }

  collectives::TensorInfo TensorAllreducer::create_unified_tensor(std::vector<collectives::TensorInfo> const& tensor_infos)
  {
    const collectives::GradID unified_tensor_id = 1UL;
    std::size_t nelems = tensor_infos.front().get_nelems();
    collectives::BufferElementType elem_type = tensor_infos.front().get_elem_type();

    tensor_sizes_bytes.emplace_back(tensor_infos.front().get_nelems()
                                    * getDataTypeSize(tensor_infos.front().get_elem_type()));

    for(auto i = 1UL; i < tensor_infos.size(); ++i)
    {
      if (elem_type != tensor_infos[i].get_elem_type())
      {
        throw std::logic_error("TensorAllreducer::create_unified_tensor: Tensors need to have same data type");
      }

      nelems += tensor_infos[i].get_nelems();
      tensor_sizes_bytes.emplace_back(tensor_infos[i].get_nelems()
                                      * getDataTypeSize(tensor_infos[i].get_elem_type()));
    }

    return {unified_tensor_id,
            nelems,
            elem_type};
  }

  template <typename AllreduceAlgorithm>
  void TensorAllreducer::create_resources_and_allreduce_op(collectives::TensorInfo const& tensor_info,
                                                           GPI::SegmentID const& segment_id)
  {
    auto const overhead_factor = 3.5;
    auto const segment_size = tensor_info.get_nelems()
                              * getDataTypeSize(tensor_info.get_elem_type())
                              * overhead_factor;

    resource_manager.make_segment_resources(segment_id, group, segment_size);
    barrier.blocking_barrier();

    collectives::Allreduce::ResourceList resources;
    auto const required_resources = AllreduceAlgorithm::get_required_resources(tensor_info,
                                                                                group);
    for (auto const& [segment_type, resource] : required_resources)
    {
      resources.emplace(
          segment_type,
          collectives::Allreduce::Resource(
            resource_manager.get_buffer_of_size(segment_id, resource.get_buffer_size_bytes()),
            resource_manager.get_notification_range(segment_id, resource.get_num_notifications())
          )
        );
    }

    allreduce_op = std::make_unique<AllreduceAlgorithm>(tensor_info,
                                                        reduction_op,
                                                        resources,
                                                        queue_handler,
                                                        group);
  }

  void TensorAllreducer::exec_allreduce(std::vector<const void*> data_ptrs,
                                        std::vector<void*> output_ptrs)
  {
    for (std::size_t i = 0; i < data_ptrs.size(); ++i)
    {
      copy_data_to_segment(data_ptrs[i], tensor_sizes_bytes[i]);
    }

    allreduce_op->start();
    while (!allreduce_op->is_finished())
    {
      allreduce_op->trigger_communication_step();
    }

    for (std::size_t i = 0; i < output_ptrs.size(); ++i)
    {
      copy_data_from_segment(output_ptrs[i], tensor_sizes_bytes[i]);
    }
    barrier.blocking_barrier();
  }

  void TensorAllreducer::copy_data_to_segment(const void* data_ptr, std::size_t size)
  {
    auto const segment_ptr = reinterpret_cast<void*>(allreduce_op->get_input_ptr());
    std::memcpy(segment_ptr, data_ptr, size);
  }

  void TensorAllreducer::copy_data_from_segment(void* result_ptr, std::size_t size)
  {
    auto const segment_ptr = reinterpret_cast<void*>(allreduce_op->get_result_ptr());
    std::memcpy(result_ptr, segment_ptr, size);
  }
}