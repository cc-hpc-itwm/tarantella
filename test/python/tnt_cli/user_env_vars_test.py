import pytest
import shlex

import runtime.platform_config as platform_config
import runtime.tarantella_cli as cli

class TestsTarantellaCLIEnvironmentVariables:

  test_cases_command_line = [
    (# CLI args
     '-x VAR0=1 -- dummy.py',
     # expected dictionary of key=values to be exported
     {'VAR0':'1'}
    ),
    (# CLI args
     '-x VAR0=value -- dummy.py',
     # expected dictionary of key=values to be exported
     {'VAR0':'value'}
    ),
    (# CLI args
     '-x VAR0= -- dummy.py',
     # expected dictionary of key=values to be exported
     {'VAR0':''}
    ),
    (# CLI args
     '-x env1="value with spaces" -- dummy.py',
     # expected dictionary of key=values to be exported
     {'env1':'value with spaces'}
    ),
    (# CLI args
     '-x env1="value with=equal sign" -- dummy.py',
     # expected dictionary of key=values to be exported
     {'env1':'value with=equal sign'}
    ),
    (# CLI args
     '-x ENV0=100 ENV1=value -- dummy.py',
     # expected dictionary of key=values to be exported
     {'ENV0':'100', 'ENV1':'value'}
    ),
    (# CLI args
     '-x ENV0="some name" ENV1="other name" -- dummy.py',
     # expected dictionary of key=values to be exported
     {'ENV0':'some name', 'ENV1':'other name'}
    ),
    (# CLI args
     '-n 4 -x RANK=$GASPI_RANK PATH_TO_BOOST=/scratch/boost test_name="My Test" --hostfile ./hostfile -- path/to/my/model.py',
     # expected dictionary of key=values to be exported
     {'RANK':'$GASPI_RANK', 'PATH_TO_BOOST':'/scratch/boost', 'test_name':'My Test'}
    ),
  ]

  @pytest.mark.parametrize("test_case", test_cases_command_line)
  def test_cli_set_environment_variable(self, test_case):
    args_string, expected_exports = test_case

    parser = cli.create_parser()
    args = parser.parse_args(shlex.split(args_string))
    tarantella = cli.TarantellaCLI(platform_config.generate_nodes_list(), 1, 1, args)

    executable_script = tarantella.generate_executable_script()
    contents = executable_script.get_initial_contents().split('\n')
    for key, value in expected_exports.items():
      assert f"export {key}=\"{value}\"" in contents

  @pytest.mark.parametrize("args_string",
                           ['-x ENV1=100 path=/my/path another_path="another path" -- dummy.py'])
  def test_cli_set_environment_variables_preserve_order(self, args_string):
    parser = cli.create_parser()
    args = parser.parse_args(shlex.split(args_string))
    tarantella = cli.TarantellaCLI(platform_config.generate_nodes_list(), 1, 1, args)

    executable_script = tarantella.generate_executable_script()
    contents = executable_script.get_initial_contents().split('\n')

    expected_exports_list = ['export ENV1="100"',
                             'export path="/my/path"',
                             'export another_path="another path"']
    found_exports = []
    for line in contents:
      if line in expected_exports_list:
        found_exports += [line]
    assert found_exports == expected_exports_list

  error_test_cases = ['-x ENV0 -- dummy.py',
                      '-x ENV0 100 -- dummy.py',
                      '-x ENV0=100 ENV2 -- dummy.py',
                      ]
  @pytest.mark.parametrize("args_string", error_test_cases)
  def test_cli_set_environment_variables_wrong_format(self, args_string):
    parser = cli.create_parser()
    args = parser.parse_args(shlex.split(args_string))
    with pytest.raises(ValueError):
      tarantella = cli.TarantellaCLI(platform_config.generate_nodes_list(), 1, 1, args)
