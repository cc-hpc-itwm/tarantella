import tarantella as tnt

import numpy as np
import pytest

# Assumption: the partitioning graph is oriented,
# such that each rank receives data from smaller ranks and sends to larger ranks
partition_test_cases = [
  # Test case 0:
  # (rank) --conn_id-- (rank)
  # (0) --0--> (1) --1--> (2) --2--> (3)
  { # conn_id, <<rank, rank>,  size>
            0: ((0,       1),     4),
            1: ((1,       2),    32),
            2: ((2,       3),   100),
  },
  # Test case 1:
  # (0) --3--> (2)
  #             ^
  #             |
  #             2
  #             |
  #            (1) --1--> (3)
  { # conn_id, <<rank, rank>,  size>
            1: ((3,       1),  1000),
            2: ((1,       2),    64),
            3: ((2,       0),    68),
  },
  # Test case 2:
  # (0) --0--> (3)
  # (1) --1--> (3)
  # (2) --2--> (3)
  { # conn_id, <<rank, rank>,  size>
            1: ((0,       3),     4),
            2: ((1,       3),     4),
            3: ((2,       3),     8),
  },
  # Test case 3:
  # (0) --0--> (3)
  # (1)
  # (2)
  { # conn_id, <<rank, rank>,  size>
            0: ((0,       3),     4),
  },
  ]

def get_other_rank(partition, conn_id):
  ranks_in_conn = partition[conn_id][0]
  other_rank = ranks_in_conn[0] if ranks_in_conn[1] == tnt.get_rank() else ranks_in_conn[1]
  return other_rank

@pytest.mark.tfversion(['2.2', '2.3', '2.4'])
class TestPipelineCommunicator:

  @pytest.mark.parametrize("partition", partition_test_cases)
  @pytest.mark.parametrize("num_micro_batches", [1,2,3])
  def test_send_all_connections(self, partition, num_micro_batches):
    elem_type = np.dtype(np.float32)
    micro_batch_size = 1
    pipeline_comm = tnt.PipelineCommunicator(partition, micro_batch_size, num_micro_batches)
        
    # send on all connections
    for micro_batch_id in range(num_micro_batches):
      for conn_id in pipeline_comm.get_local_connection_ids():
        if get_other_rank(partition, conn_id) < tnt.get_rank():
          continue

        array_length = partition[conn_id][1] // elem_type.itemsize
        input_array = np.empty(shape=(array_length, 1), dtype=elem_type)
        input_array.fill(tnt.get_rank())

        pipeline_comm.send(input_array,
                           connection_id = conn_id,
                           micro_batch_id = micro_batch_id)

    # receive on all connections
    for micro_batch_id in range(num_micro_batches):
      for conn_id in pipeline_comm.get_local_connection_ids():
        if get_other_rank(partition, conn_id) > tnt.get_rank():
          continue

        array_length = partition[conn_id][1] // elem_type.itemsize
        expected_array = np.empty(shape=(array_length, 1), dtype=elem_type)
        expected_array.fill(get_other_rank(partition, conn_id))

        input_array = np.empty(shape=(array_length, 1))
        result = pipeline_comm.recv(input_array,
                                    connection_id = conn_id,
                                    micro_batch_id = micro_batch_id)
        assert np.allclose(result, expected_array, atol=1e-6)

