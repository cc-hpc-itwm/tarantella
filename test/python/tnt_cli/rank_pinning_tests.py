import logging
import numpy as np
import pytest
import shutil

import runtime.platform_config as platform_config
import runtime.tarantella_cli as cli

@pytest.mark.skipif(shutil.which("numactl") is None,
                    reason="`numactl` has to be installed/added to PATH")
class TestCLIpinToSocket:

  @pytest.fixture(params=[1, 2])
  def npernode(self, request):
    nnp = request.param
    if nnp > cli.get_numa_nodes_count():
        pytest.skip()
    return nnp

  def test_pin_to_socket(self, npernode):
    parser = cli.create_parser()
    args = parser.parse_args(["-n", f"{npernode}", "--pin-to-socket", "--", "dummy.py"])

    tarantella = cli.TarantellaCLI(platform_config.generate_nodes_list(),
                                   num_gpus_per_node = 0, num_cpus_per_node = npernode,
                                   args = args)
    executable_script = tarantella.generate_executable_script()
    contents = executable_script.get_initial_contents().split('\n')
    assert f"socket=$(( $GASPI_RANK % {npernode} ))" in contents
    assert f"numactl --cpunodebind=$socket --membind=$socket python dummy.py" in contents

  @pytest.mark.parametrize("npernode", [1000])
  def test_pin_more_ranks_than_sockets(self, npernode):
    parser = cli.create_parser()
    args = parser.parse_args(["-n", f"{npernode}", "--pin-to-socket", "--", "dummy.py"])

    with pytest.raises(ValueError):
      cli.TarantellaCLI(platform_config.generate_nodes_list(),
                        num_gpus_per_node = 0, num_cpus_per_node = npernode,
                        args = args)
