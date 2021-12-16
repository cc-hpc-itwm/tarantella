#include "PipelineCommunicator.hpp"
#include "GlobalContextFixture.hpp"

#include <GaspiCxx/Runtime.hpp>

#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>

#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace tarantella
{
  using LayerEdges = PipelineCommunicator::LayerEdges;
  using ConnectionID = std::size_t;
  using Rank = gaspi::group::GlobalRank;
  using MessageSizeBytes = std::size_t;
  using T = int;

  // list of partitions for each rank
  using PartitionTestCase = std::pair<std::size_t, std::vector<LayerEdges>>;

  template <typename T>
  class Partition
  {
    public:
      explicit Partition(PartitionTestCase test_case, Rank current_rank,
                         MessageSizeBytes micro_batch_size)
      : edges(test_case.second[current_rank]),
        num_microbatches(test_case.first),
        micro_batch_size(micro_batch_size)
      { }

      auto get_num_microbatches()
      {
        return num_microbatches;
      }

      auto get_edges()
      {
        return edges;
      }

      std::size_t get_num_elements(ConnectionID const connection_id)
      {
        if (edges.find(connection_id) == edges.end())
        {
          throw std::runtime_error("Unknown connection id");
        }
        if (edges[connection_id].second % sizeof(T) != 0)
        {
          throw std::runtime_error("Buffer size is not a multiple of type size");
        }
        return edges[connection_id].second  * micro_batch_size / sizeof(T);
      }

      Rank get_other_rank(ConnectionID const connection_id)
      {
        if (edges.find(connection_id) == edges.end())
        {
          throw std::runtime_error("Unknown connection id");
        }
        return edges[connection_id].first;
      }

      auto get_connection_ids()
      {
        std::vector<ConnectionID> connection_ids;
        for (auto& edge : edges)
        {
          connection_ids.push_back(edge.first);
        }
        return connection_ids;
      }

    private:
      LayerEdges edges;
      std::size_t num_microbatches;
      MessageSizeBytes micro_batch_size;
  };
}

namespace std
{
  std::ostream& operator<< (std::ostream& os, const tarantella::PartitionTestCase&)
  {
    os << std::endl;
    return os;
  }
}

namespace tarantella
{
  BOOST_GLOBAL_FIXTURE(GlobalContext);

  // Assumption: the partitioning graph is oriented,
  // such that each rank receives data from smaller ranks and sends to larger ranks
  std::vector<PartitionTestCase> partition_test_cases = {
    // Test case 1:
    // (rank) --conn_id-- (rank)
    // (0) --0--> (1) --1--> (2) --2--> (3)
    { 2, // num microbatches
      { // conn_id, <rank,  size>
          {{0,         {1,   100}}  // rank 0
          },
          {{0,         {0,   100}}, // rank 1
           {1,         {2,   300}}
          },
          {{1,         {1,   300}}, // rank 2
           {2,         {3,     4}}
          },
          {{2,         {2,     4}}  // rank 3
          }
      }
    },
   // Test case 2
   // (0) --3--> (2)
   //             ^
   //             |
   //             2
   //             |
   //            (1) --1--> (3)
    { 3, // num microbatches
      { // conn_id, <rank,  size>
          {{3,         {2,     4}}  // rank 0
          },
          {{2,         {2,  1600}}, // rank 1
           {1,         {3,   100}}
          },
          {{3,         {0,     4}}, // rank 2
           {2,         {1,  1600}}
          },
          {{1,         {1,   100}}  // rank 3
          }
      }
    },
    // Test case 3
    { 1, // num microbatches
      { // conn_id, <rank,  size>
          {{1,         {2,  40000}},  // rank 0
          },
          {},                        // rank 1
          {{1,         {0,  40000}}, // rank 2
          },
          {}                         // rank 3
      }
    }
  };

  BOOST_AUTO_TEST_SUITE(pipelinecomm_send_recv_unit)
    BOOST_AUTO_TEST_CASE(check_number_of_ranks)
    {
      BOOST_TEST_REQUIRE(gaspi::getRuntime().size() == 4);
    }

    BOOST_DATA_TEST_CASE(pipelinecomm_send_all_connections, partition_test_cases, test_case)
    {
      MessageSizeBytes micro_batch_size = 16;
      auto rank = gaspi::getRuntime().global_rank();
      Partition<T> partition(test_case, rank, micro_batch_size);
      auto pipeline_comm = PipelineCommunicator(partition.get_edges(),
                                                partition.get_num_microbatches());
      pipeline_comm.setup_infrastructure(micro_batch_size);

      // send from all connections
      for (auto connection_id : partition.get_connection_ids())
      {
        if (partition.get_other_rank(connection_id) < rank)
        {
          continue;
        }
        std::vector<T> send_buffer(partition.get_num_elements(connection_id));
        std::iota(send_buffer.begin(), send_buffer.end(), rank);

        for (auto mbatch_id = 0UL; mbatch_id < partition.get_num_microbatches(); ++mbatch_id)
        {
          pipeline_comm.send(send_buffer.data(), connection_id, mbatch_id);
        }
      }

      // receive on all connections
      for (auto mbatch_id = 0UL; mbatch_id < partition.get_num_microbatches(); ++mbatch_id)
      {
        for (auto connection_id : partition.get_connection_ids())
        {
          if (partition.get_other_rank(connection_id) > rank)
          {
            continue;
          }
          std::vector<T> expected_buffer(partition.get_num_elements(connection_id));
          std::iota(expected_buffer.begin(), expected_buffer.end(),
                    partition.get_other_rank(connection_id));

          std::vector<T> recv_buffer(partition.get_num_elements(connection_id));
          pipeline_comm.recv(recv_buffer.data(), connection_id, mbatch_id);
          BOOST_TEST_REQUIRE(recv_buffer == expected_buffer);
        }
      }
    }

  BOOST_AUTO_TEST_SUITE_END()
}
