#include "utils.hpp"

namespace tarantella
{
  namespace collectives
  {
    namespace Allreduce
    {
      allreduce::dataType to_allreduce_dataType(const BufferElementType type)
      {
        std::unordered_map<BufferElementType, allreduce::dataType> const types{
            {BufferElementType::FLOAT, allreduce::FLOAT},
            {BufferElementType::DOUBLE, allreduce::DOUBLE},
            {BufferElementType::INT16, allreduce::INT16},
            {BufferElementType::INT32, allreduce::INT32},
        };
        return types.at(type);
      }

      allreduce::reductionType to_allreduce_reductionType(const Operator::ReductionOp op)
      {
        std::unordered_map<Operator::ReductionOp, allreduce::reductionType> const reduction_ops{
            {Operator::ReductionOp::SUM, allreduce::SUM},
            {Operator::ReductionOp::AVERAGE, allreduce::AVERAGE},
        };
        return reduction_ops.at(op);
      }

      allreduceButterfly::segmentBuffer to_allreduce_segment_buffer(Operator::Resource const& resource)
      {
        auto const [data_segment_buffer, notif_range] = resource;
        allreduceButterfly::segmentBuffer buffer{data_segment_buffer.get_segment_id(),
                                                 data_segment_buffer.get_offset(),
                                                 static_cast<gaspi_notification_id_t>(notif_range.first)};
        return buffer;
      }
    }
  }
}
