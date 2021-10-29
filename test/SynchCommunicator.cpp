#include "BufferElementType.hpp"
#include "SynchCommunicator.hpp"
#include "utilities.hpp"
#include "GlobalContextFixture.hpp"

#include <GaspiCxx/Runtime.hpp>

#include <iostream>
#include <numeric>
#include <future>

#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>

using boost::test_tools::per_element;

namespace tarantella
{
  BOOST_GLOBAL_FIXTURE(GlobalContext);
  float const epsilon_f(1e-6);

  std::vector<std::vector<collectives::TensorInfo>> test_cases
  {
    { 
      // test case #1
      // (tensor_id, num_elements, element_type)
      {1, 8, collectives::BufferElementType::FLOAT}
    },
    { 
      // test case #2
      {5, 4 * 100 , collectives::BufferElementType::FLOAT}
    },
    { 
      // test case #3
      {42, 17, collectives::BufferElementType::FLOAT},
      {11, 23, collectives::BufferElementType::FLOAT},
    },
    {
      // test case #4
      {1, 8, collectives::BufferElementType::FLOAT},
      {2, 8, collectives::BufferElementType::FLOAT},
      {3, 8, collectives::BufferElementType::FLOAT},
      {4, 8, collectives::BufferElementType::FLOAT},
      {5, 8, collectives::BufferElementType::FLOAT},
      {6, 9, collectives::BufferElementType::FLOAT},
    },
    {
      // test case #5
      {1, 412, collectives::BufferElementType::FLOAT},
      {2, 500, collectives::BufferElementType::FLOAT},
      {3, 612, collectives::BufferElementType::FLOAT},
      {4,  17, collectives::BufferElementType::FLOAT},
      {5, 800, collectives::BufferElementType::FLOAT},
      {6, 945, collectives::BufferElementType::FLOAT},
    },
  };

  std::vector<std::size_t> thresholds_bytes
  {
    0UL,
    4UL,
    64UL,
    1024UL,
  };

  class SynchCommTestData
   {
      public:

        SynchCommTestData(std::vector<collectives::TensorInfo> const& tensor_infos, 
                          gaspi::collectives::ReductionOp op,
                          std::size_t threshold_bytes = 0UL)
        : group(),
          synch_comm(group,
                     tensor_infos,
                     threshold_bytes),
          expected_output_bufs(tensor_infos.size()),
          input_bufs(tensor_infos.size()),
          op(op)
        {
          auto nranks = gaspi::getRuntime().size();
          auto rank = gaspi::getRuntime().global_rank();

          // generate data for each tensor and fill in the expected result after Allreduce
          for (auto grad_idx = 0U; grad_idx < tensor_infos.size(); ++grad_idx)
          {
            ids.push_back(tensor_infos.at(grad_idx).get_id());

            // create input buffers for Allreduce based on the buffer size specified in the test case
            std::generate_n(std::back_inserter(input_bufs[grad_idx]), 
                            tensor_infos.at(grad_idx).get_nelems(), 
                            [&]() 
                            { 
                              // fill buffer with values based on element index and current rank
                              auto idx = input_bufs[grad_idx].size(); 
                              return idx * (rank + 1); 
                            });

            // create expected result buffers and fill them according to the tested Allreduce operation
            std::generate_n(std::back_inserter(expected_output_bufs[grad_idx]), 
                            tensor_infos.at(grad_idx).get_nelems(), 
                            [&]() 
                            { 
                              auto idx = expected_output_bufs[grad_idx].size(); 
                              auto elem = -1.f;
                              switch (op)
                              {
                                case gaspi::collectives::ReductionOp::SUM:
                                {
                                  elem = idx * nranks * (nranks + 1.) / 2.;
                                  break;
                                }
                                default:
                                {
                                  throw std::runtime_error(
                                    "[Test][SynchCommunicator] Unknown reduction operation");
                                }
                              }
                              return elem;
                            });
          }
        };

        int get_index_for_id(tarantella::GradID id) 
        {
          auto it = std::find(ids.begin(), ids.end(), id);
          if (it == ids.end())
          {
            throw std::invalid_argument("ID not found in the list of ids for the current test case");
          }
          return distance(ids.begin(), it);
        }

        gaspi::group::Group group;
        tarantella::SynchCommunicator synch_comm;
        std::vector<std::vector<float> > expected_output_bufs;
        std::vector<std::vector<float> > input_bufs;
        std::vector<tarantella::GradID> ids;
        gaspi::collectives::ReductionOp const op;
  };

  BOOST_AUTO_TEST_SUITE(synch_communicator_unit)

    BOOST_DATA_TEST_CASE(synch_comm_creation, test_cases, test_case)
    {
      BOOST_REQUIRE_NO_THROW(SynchCommunicator synch_comm(gaspi::group::Group(),
                                                          test_case));
    }

    BOOST_TEST_DECORATOR(*boost::unit_test::tolerance(epsilon_f));
    BOOST_DATA_TEST_CASE(synch_comm_serialized_allred, test_cases * thresholds_bytes, test_case, threshold) // Cartesian product
    {
      auto const op = gaspi::collectives::ReductionOp::SUM;
      SynchCommTestData synch_comm_data(test_case, op, threshold);

      for (auto &id : synch_comm_data.ids)
      {
        auto input_buf = synch_comm_data.input_bufs.at(synch_comm_data.get_index_for_id(id));
        synch_comm_data.synch_comm.start_allreduce_impl(id, input_buf.data());
      }

      for (GradID &id : synch_comm_data.ids)
      {
        std::vector<float> out_data(synch_comm_data.expected_output_bufs.at(synch_comm_data.get_index_for_id(id)).size());
        synch_comm_data.synch_comm.finish_allreduce_impl(id, out_data.data());
        BOOST_TEST_REQUIRE(out_data == synch_comm_data.expected_output_bufs.at(synch_comm_data.get_index_for_id(id)), per_element());
      }
    }

    namespace
    {
      void execute_iteration(SynchCommTestData& synch_comm_data)
      {
        std::vector<std::future<GradID>> futures;

        // create multiple allreduce calls in parallel
        for (auto &id : synch_comm_data.ids)
        {
          futures.emplace_back(std::async(
              std::launch::async,
              [&synch_comm_data](const GradID id) -> GradID {
                auto input_buf = synch_comm_data.input_bufs.at(synch_comm_data.get_index_for_id(id));
                synch_comm_data.synch_comm.start_allreduce_impl(id, input_buf.data());
                return id;
              },
              id));
        }
        // wait for all allreduce operations to be submitted
        for (auto &f : futures)
        {
          f.get();
        }
        // wait for the execution of each allreduce to end and verify result
        for (auto& id : synch_comm_data.ids)
        {
          std::vector<float> out_data(synch_comm_data.expected_output_bufs.at(synch_comm_data.get_index_for_id(id)).size());
          synch_comm_data.synch_comm.finish_allreduce_impl(id, out_data.data());
          BOOST_TEST_REQUIRE(out_data == synch_comm_data.expected_output_bufs.at(synch_comm_data.get_index_for_id(id)), 
                             per_element());
        }
      }
    }

    BOOST_TEST_DECORATOR(*boost::unit_test::tolerance(epsilon_f));
    BOOST_DATA_TEST_CASE(synch_comm_parallel_allred, test_cases * thresholds_bytes, test_case, threshold)
    {
      auto const op = gaspi::collectives::ReductionOp::SUM;
      SynchCommTestData synch_comm_data(test_case, op, threshold);
      execute_iteration(synch_comm_data);
    }

    BOOST_TEST_DECORATOR(*boost::unit_test::tolerance(epsilon_f));
    BOOST_DATA_TEST_CASE(synch_comm_repeat_parallel_allred, test_cases * thresholds_bytes, test_case, threshold)
    {
      auto const op = gaspi::collectives::ReductionOp::SUM;
      auto nreps = 10UL;
      SynchCommTestData synch_comm_data(test_case, op, threshold);
      for (auto i = 0UL; i < nreps; ++i)
      {
        execute_iteration(synch_comm_data);
      }
    }

  BOOST_AUTO_TEST_SUITE_END()
}
