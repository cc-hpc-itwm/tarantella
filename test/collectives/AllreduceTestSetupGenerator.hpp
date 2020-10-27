#pragma once

#include "allreduceButterfly.h"
#include "gpi/Context.hpp"

#include <algorithm>
#include <iostream>
#include <vector>

#include <boost/test/unit_test.hpp>

using TestCase = std::vector<std::vector<float>>;
namespace std
{
  std::ostream &operator<<(std::ostream &os, TestCase const &test)
  {
    for (auto i = 0U; i < test.size(); ++i)
    {
      os << "Data for rank " << i << "/" << test.size() << ": [";
      for (auto const elem : test[i])
      {
        os << elem << " ";
      }
      os << "]" << std::endl;
    }
    return os;
  }
}
namespace tarantella
{
  using AllreduceDataType = collectives::allreduce::dataType;
  using AllreduceOp = collectives::allreduce::reductionType;
  // create expected allreduce results buffers for each test case 
  template<AllreduceDataType T, AllreduceOp op>
  class AllreduceTestSetupGenerator
  {
    // determine the Allreduce element types based on the datatype template parameter
    using BufferType = typename std::conditional<T == AllreduceDataType::INT32, int32_t,
                       typename std::conditional<T == AllreduceDataType::INT16, int16_t,
                       typename std::conditional<T == AllreduceDataType::DOUBLE, double,
                       float>::type >::type >::type;

    public:
      AllreduceTestSetupGenerator(tarantella::GPI::Context& ctx, TestCase const& data,
                                  tarantella::GPI::SegmentID data_segment_id,
                                  tarantella::GPI::SegmentID comm_segment_id,
                                  gaspi_notification_id_t first_notification_id)
          : context(ctx),
            first_notification_id(first_notification_id),
            group_size(data.size()),
            data_seg_buffer({data_segment_id, offset, first_notification_id}),
            comm_seg_buffer({comm_segment_id, offset, first_notification_id}),
            input_buf(generate_rank_input_buf(data, context.get_rank())),
            expected_output_buf(generate_expected_output_buf(data, op))
      {}
      virtual ~AllreduceTestSetupGenerator() = default;

      void copy_data_to_segment(void* seg_ptr)
      {
        std::memcpy(seg_ptr, input_buf.data(), input_buf.size()*sizeof(BufferType));
      }

      std::vector<BufferType> copy_results_from_segment(void* seg_ptr)
      {
        std::vector<BufferType> output_buf(input_buf.size());
        std::memcpy(output_buf.data(), seg_ptr, input_buf.size()*sizeof(BufferType));
        return output_buf;
      }

      AllreduceDataType get_elem_type() const {return T;};

      GPI::Context& context;
      size_t const offset = 0;
      gaspi_notification_id_t const first_notification_id;
      size_t group_size;

      collectives::allreduceButterfly::segmentBuffer data_seg_buffer;
      collectives::allreduceButterfly::segmentBuffer comm_seg_buffer;
      collectives::queues queue_handler;
      std::vector<BufferType> input_buf;
      std::vector<BufferType> expected_output_buf;

    private:

      std::vector<BufferType> generate_rank_input_buf(TestCase const& data, gaspi_rank_t const rank)
      {
        std::vector<BufferType> in_buf;
        BOOST_TEST_REQUIRE(rank < group_size);
        std::transform(data[rank].begin(), data[rank].end(),
                       std::back_inserter(in_buf),
                       [](auto elem) { return static_cast<BufferType>(elem); });
        return in_buf;
      }

      std::vector<BufferType> generate_expected_output_buf(TestCase const& data,
          AllreduceOp operation)
      {
        std::vector<BufferType> out_buf;
        switch (operation)
        {
          case AllreduceOp::SUM:
          {
            out_buf = compute_sum_over_ranks(data);
            break;
          }
          case AllreduceOp::AVERAGE:
          {
            out_buf = compute_sum_over_ranks(data);
            std::transform(out_buf.begin(), out_buf.end(),
                           out_buf.begin(),
                           [group_size=group_size](auto elem) { return elem/static_cast<BufferType>(group_size);}
                           );
            break;
          }
          default:
          {
            throw std::runtime_error("[AllreduceTestSetupGenerator] Unknown reduction operation");
          }
        }
        return out_buf;
      }

      std::vector<BufferType> compute_sum_over_ranks(TestCase const& data)
      {
        std::vector<BufferType> out_buf(data.front().size());
        for (auto const& buffer : data)
        {
          std::transform(buffer.begin(), buffer.end(), out_buf.begin(),
                         out_buf.begin(),
                         [](auto elem1, auto elem2) {
                            return static_cast<BufferType>(elem1) + static_cast<BufferType>(elem2);}
                         );
        }
        return out_buf;
      }
  };

  template <AllreduceDataType T, AllreduceOp op>
  class AllreduceDoubleBufferTestSetupGenerator : public AllreduceTestSetupGenerator<T, op>
  {

    public:
      AllreduceDoubleBufferTestSetupGenerator(tarantella::GPI::Context& ctx, TestCase const& data,
                                              tarantella::GPI::SegmentID data_segment_id0,
                                              tarantella::GPI::SegmentID data_segment_id1,
                                              tarantella::GPI::SegmentID comm_segment_id,
                                              gaspi_notification_id_t first_notification_id)
          : AllreduceTestSetupGenerator<T, op>(ctx, data, 
                                               data_segment_id0, comm_segment_id,
                                               first_notification_id),
            additional_data_seg_buffer({data_segment_id1, this->offset, this->first_notification_id})
      {}
      virtual ~AllreduceDoubleBufferTestSetupGenerator() = default;

      collectives::allreduceButterfly::segmentBuffer additional_data_seg_buffer;
  };
}
