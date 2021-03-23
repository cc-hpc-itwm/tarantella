#include "PipelineCommunicator.hpp"
#include "GlobalContextFixture.hpp"

#include <GaspiCxx/Runtime.hpp>

#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>

#include <numeric>
#include <unordered_map>
#include <vector>

namespace tarantella
{
  struct PartitionInfo
  {
    std::size_t conn_id;
    std::size_t num_micro_batches;
    std::size_t num_elements;
  };
}

namespace std
{
  std::ostream& operator<< (std::ostream& os, const tarantella::PartitionInfo& pinfo)
  {
    os << "ConnID=" << pinfo.conn_id << " num_micro_batches=" << pinfo.num_micro_batches
        << " num_elements=" << pinfo.num_elements << std::endl;
    return os;
  }
}

namespace tarantella
{
  BOOST_GLOBAL_FIXTURE(GlobalContext);

  using LayerEdges = PipelineCommunicator::LayerEdges;
  using Rank = gaspi::group::GlobalRank;
  using T = int;

  // list of partition details corresponding to each rank
  // (both ranks have identical Edges lists)
  std::vector<PartitionInfo> partition_info_test_cases =
                            //connID num_micro_batches num_elements
                            { { 3,                  1,         1  },
                              { 0,                  1,         0  },
                              { 5,                  5,         8  },
                              { 0,                  2,       300  }
                            };

  BOOST_AUTO_TEST_SUITE(pipelinecomm_send_recv_unit)
    BOOST_AUTO_TEST_CASE(check_number_of_ranks)
    {
      BOOST_TEST_REQUIRE(gaspi::getRuntime().size() == 2);
    }

    BOOST_AUTO_TEST_CASE(pipelinecomm_initialize)
    {
      auto num_micro_batches = 0UL;
      BOOST_REQUIRE_NO_THROW(PipelineCommunicator({}, num_micro_batches));
    }

    BOOST_DATA_TEST_CASE(pipelinecomm_send_recv, partition_info_test_cases, partition_info)
    {
      auto rank = gaspi::getRuntime().global_rank();
      Rank other_rank = (rank == 0) ? 1 : 0;

      std::vector<T> send_buffer(partition_info.num_elements);
      std::iota(send_buffer.begin(), send_buffer.end(), 42);

      auto conn_id = partition_info.conn_id;
      auto mbatch_id = partition_info.num_micro_batches - 1;  // use last micro-batch ID
      LayerEdges partition = {{ conn_id, { other_rank,
                                           partition_info.num_elements * sizeof(T) } }
                             };
      auto pipeline_comm = PipelineCommunicator(partition,
                                                partition_info.num_micro_batches);
      if (rank == 0)
      {
        pipeline_comm.send(send_buffer.data(), conn_id, mbatch_id);
      }
      else if (rank == 1)
      {
        std::vector<T> recv_buffer(partition_info.num_elements);

        pipeline_comm.recv(recv_buffer.data(), conn_id, mbatch_id);
        BOOST_TEST_REQUIRE(recv_buffer == send_buffer);
      }
    }

    BOOST_DATA_TEST_CASE(pipelinecomm_ping_pong, partition_info_test_cases, partition_info)
    {
      auto rank = gaspi::getRuntime().global_rank();
      Rank other_rank = (rank == 0) ? 1 : 0;

      std::vector<T> send_buffer_0to1(partition_info.num_elements);
      std::vector<T> send_buffer_1to0(partition_info.num_elements);
      std::iota(send_buffer_0to1.begin(), send_buffer_0to1.end(), 42);
      std::iota(send_buffer_1to0.begin(), send_buffer_1to0.end(), 2);

      auto conn_id = partition_info.conn_id;
      auto mbatch_id = partition_info.num_micro_batches / 2;  // use median micro-batch ID
      LayerEdges partition = {{ conn_id, { other_rank,
                                           partition_info.num_elements * sizeof(T) } }
                             };
      auto pipeline_comm = PipelineCommunicator(partition, partition_info.num_micro_batches);
      if (rank == 0)
      {
        std::vector<T> recv_buffer(partition_info.num_elements);
        pipeline_comm.send(send_buffer_0to1.data(), conn_id, mbatch_id);

        pipeline_comm.recv(recv_buffer.data(), conn_id, mbatch_id);
        BOOST_TEST_REQUIRE(recv_buffer == send_buffer_1to0);
      }
      else if (rank == 1)
      {
        std::vector<T> recv_buffer(partition_info.num_elements);
        pipeline_comm.recv(recv_buffer.data(), conn_id, mbatch_id);
        BOOST_TEST_REQUIRE(recv_buffer == send_buffer_0to1);

        pipeline_comm.send(send_buffer_1to0.data(), conn_id, mbatch_id);
      }
    }

  BOOST_AUTO_TEST_SUITE_END()
}
