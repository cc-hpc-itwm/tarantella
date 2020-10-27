#include "AllreduceTestSetupGenerator.hpp"
#include "allreduceButterflyDoubleBuffer.h"
#include "collectives/barrier/GPIBarrier.hpp"
#include "GlobalContextFixture.hpp"
#include "gpi/ResourceManager.hpp"

#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/mpl/list.hpp>

using boost::test_tools::per_element;

namespace tarantella
{
  namespace
  {
    std::vector<gaspi_rank_t> gen_group_ranks(size_t nranks_in_group)
    {
      std::vector<gaspi_rank_t> group_ranks(nranks_in_group);
      std::iota(group_ranks.begin(), group_ranks.end(), 0);
      return group_ranks;
    }
  }
  
  BOOST_GLOBAL_FIXTURE( GlobalContext );

  float const epsilon_f(1e-6);
  double const epsilon_d(1e-12);

  // Test cases defining input buffers for Allreduce on a number of ranks given by 
  // the number of buffers in each test case 
  std::vector<TestCase> test_cases
  {
    { // test case #1 (nelems = 0)
        {}  // rank0
    },
    { // test case #2 (nelems = 1)
        {1}  // rank0
    },
    { // test case #3
        {2.34, 3, 4, 5, 6}  // rank0
    },
    { // test case #4  (nelems > nranks, nelems%nranks == 0)
        {1, 2, 3, 0.8},  // rank0
        {0.1, 0.2, 5, 6} // rank1
    },
  //  { // test case #5  (nelems = 1)
  //         {2,3},  // rank0
  //         {3,4},  // rank1
  //         {4,4}  // rank2
  //  },
    { // test case #6  (nelems > nranks, nelems%nranks >0)
        {1, 2, 3, 0.8},  // rank0
        {0.1, 0.2, 5, 6}, // rank1
        {0.1, 0.2, 5, 6} // rank2
    },
    { // test case #7  (nelems == nranks)
           {1, 3, 4, 5},  // rank0
           {2, 6, 77, 777},  // rank1
           {3, 42, 55, 2123},  // rank2
           {4, 423, 7,  4},  // rank3
    },
    { // test case #8 (nelems > nranks, nelems%nranks >0)
           {1, 3, 4, 5, 1},  // rank0
           {2, 6, 77, 777, 1},  // rank1
           {3, 42, 55, 2123, 1},  // rank2
           {4, 423, 7,  4, 1},  // rank3
    },
  //  { // test case #9 (nelems < nranks)
  //         {1, 3, 4},  // rank0
  //         {2, 6, 77},  // rank1
  //         {3, 42, 55},  // rank2
  //         {4, 423, 7},  // rank3
  //  },
    //  { // test case #10 (nelems = 1)
    //         {1},  // rank0
    //         {2},  // rank1
    //         {3},  // rank2
    //         {4},  // rank3
    //  },
  };

  template<AllreduceDataType datatype, AllreduceOp op>
  void exec_allreduce_double_buffer(tarantella::GPI::Context& context, TestCase const& test_case)
  {
    if (context.get_comm_size() < test_case.size())
    {
      throw std::logic_error("Allreduce test with fewer processes than required by test case");
    }

    // allocate group for the number of ranks defined by the test case
    GPI::Group const group(gen_group_ranks(test_case.size()));

    // resource configuration for the test case
    gaspi_notification_id_t const first_notification_id = 42;
    GPI::SegmentID data_segment_id0 = 1;
    GPI::SegmentID data_segment_id1 = 2;
    GPI::SegmentID comm_segment_id = 3;
    auto const data_segment_size = std::max(size_t(1), test_case[0].size() * getDataTypeSize(datatype));
    auto const comm_segment_size = std::max(size_t(1),
                                            collectives::allreduceButterfly::getNumberOfElementsSegmentCommunicate(
                                              test_case[0].size(), test_case.size()) * getDataTypeSize(datatype));

    // use new segment manager for each test case and release the resources at the end
    GPI::SegmentManager segmentmanager(context);
    
    if (group.contains_rank(context.get_rank()))
    {
      BOOST_REQUIRE_NO_THROW(segmentmanager.create_segment(data_segment_id0, group, data_segment_size));
      BOOST_REQUIRE_NO_THROW(segmentmanager.create_segment(data_segment_id1, group, data_segment_size));
      BOOST_REQUIRE_NO_THROW(segmentmanager.create_segment(comm_segment_id, group, comm_segment_size));
    }
    else
    {
      BOOST_REQUIRE_THROW(segmentmanager.create_segment(data_segment_id0, group, data_segment_size),
                          std::runtime_error);
      BOOST_REQUIRE_THROW(segmentmanager.create_segment(data_segment_id1, group, data_segment_size),
                          std::runtime_error);
      BOOST_REQUIRE_THROW(segmentmanager.create_segment(comm_segment_id, group, comm_segment_size),
                          std::runtime_error);
    }

    collectives::Barrier::GPIBarrierAllRanks barrier;
    barrier.blocking_barrier();

    if (group.contains_rank(context.get_rank()))
    { 
      // only processes in the group execute the Allreduce
      AllreduceDoubleBufferTestSetupGenerator<datatype, op> test(context, test_case, 
                                                                 data_segment_id0, data_segment_id1, 
                                                                 comm_segment_id,
                                                                 first_notification_id);
      collectives::allreduceButterflyDoubleBuffer allreduce(test.input_buf.size(),
                                                            test.get_elem_type(),
                                                            op,
                                                            test.data_seg_buffer,
                                                            test.additional_data_seg_buffer,
                                                            test.comm_seg_buffer,
                                                            test.queue_handler,
                                                            group);

      test.copy_data_to_segment(allreduce.getActiveReducePointer());
      allreduce.signal();

      while (allreduce());

      auto output_buf = test.copy_results_from_segment(allreduce.getResultsPointer());
      BOOST_TEST_REQUIRE(output_buf == test.expected_output_buf, per_element());
    }
    else // other processes should not be defined in the test case
    {
      BOOST_TEST_REQUIRE(context.get_rank() >= test_case.size());
    }

    // make sure all processes have finished before cleanup
    barrier.blocking_barrier();
  }

  BOOST_AUTO_TEST_SUITE(allreduce_butterfly_unit)

  BOOST_TEST_DECORATOR(*boost::unit_test::tolerance(epsilon_f));
  BOOST_DATA_TEST_CASE(simple_allreduce_float_sum, test_cases, test_case)
  {
    exec_allreduce_double_buffer<AllreduceDataType::FLOAT, AllreduceOp::SUM>
      (GlobalContext::instance()->gpi_cont, test_case);
  }

  BOOST_TEST_DECORATOR(*boost::unit_test::tolerance(epsilon_f));
  BOOST_DATA_TEST_CASE(simple_allreduce_float_avg, test_cases, test_case)
  {
    exec_allreduce_double_buffer<AllreduceDataType::FLOAT, AllreduceOp::AVERAGE>
      (GlobalContext::instance()->gpi_cont, test_case);
  }

  BOOST_TEST_DECORATOR(*boost::unit_test::tolerance(epsilon_d));
  BOOST_DATA_TEST_CASE(simple_allreduce_double_sum, test_cases, test_case)
  {
    exec_allreduce_double_buffer<AllreduceDataType::DOUBLE, AllreduceOp::SUM>
      (GlobalContext::instance()->gpi_cont, test_case);
  }

  BOOST_TEST_DECORATOR(*boost::unit_test::tolerance(epsilon_d));
  BOOST_DATA_TEST_CASE(simple_allreduce_double_avg, test_cases, test_case)
  {
    exec_allreduce_double_buffer<AllreduceDataType::DOUBLE, AllreduceOp::AVERAGE>
      (GlobalContext::instance()->gpi_cont, test_case);
  }

  BOOST_DATA_TEST_CASE(simple_allreduce_int32_sum, test_cases, test_case)
  {
    exec_allreduce_double_buffer<AllreduceDataType::INT32, AllreduceOp::SUM>
      (GlobalContext::instance()->gpi_cont, test_case);
  }

  BOOST_DATA_TEST_CASE(simple_allreduce_int32_avg, test_cases, test_case)
  {
    exec_allreduce_double_buffer<AllreduceDataType::INT32, AllreduceOp::AVERAGE>
      (GlobalContext::instance()->gpi_cont, test_case);
  }

  BOOST_DATA_TEST_CASE(simple_allreduce_int16_sum, test_cases, test_case)
  {
    exec_allreduce_double_buffer<AllreduceDataType::INT16, AllreduceOp::SUM>
      (GlobalContext::instance()->gpi_cont, test_case);
  }

  BOOST_DATA_TEST_CASE(simple_allreduce_int16_avg, test_cases, test_case)
  {
    exec_allreduce_double_buffer<AllreduceDataType::INT16, AllreduceOp::AVERAGE>
      (GlobalContext::instance()->gpi_cont, test_case);
  }
  BOOST_AUTO_TEST_SUITE_END()
}

