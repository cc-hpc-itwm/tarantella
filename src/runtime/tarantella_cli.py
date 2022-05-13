import argparse
import os
import shutil
import subprocess
import sys
try:
  from version import tnt_version
except:
  tnt_version = "Unknown version"

import runtime.file_management as file_man
import runtime.environment_config as env_config
from runtime import logger
from runtime import tarantella_cleanup

def create_parser():
  parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
  singlenode_group = parser.add_argument_group('Single-node execution')
  singlenode_group.add_argument("-n",
                                help="number of TensorFlow instances to start on the local node",
                                dest = "npernode_single_node",
                                metavar = "N",
                                type = int,
                                default = None)
  multinode_group = parser.add_argument_group('Multi-node execution')
  multinode_group.add_argument("--hostfile",
                               dest = "hostfile",
                               help="path to the list of nodes (hostnames) on which to execute the SCRIPT",
                               default = None)
  multinode_group.add_argument("--n-per-node", "--devices-per-node",
                               help="""number of devices (i.e., either GPUs or processes on CPUs) to be
                                       used on each node
                                    """,
                               dest = "npernode",
                               type = int,
                               default = None)
  parser.add_argument("--no-gpu", "--no-gpus",
                      help="disallow GPU usage",
                      dest = "use_gpus",
                      action='store_false',
                      default = True)
  log_levels = ('DEBUG', 'INFO', 'WARNING', 'ERROR')
  parser.add_argument('--log-level', default='WARNING', choices=log_levels,
                      help = "logging level for library messages")
  parser.add_argument("--log-on-all-devices",
                      help="enable library logging messages on all devices",
                      dest = "log_all",
                      action='store_true',
                      default = False)
  parser.add_argument("--output-on-all-devices",
                      help="enable output on all devices (e.g., training info)",
                      dest = "output_all",
                      action='store_true',
                      default = False)
  parser.add_argument("-x",
                      help = """list of space-separated KEY=VALUE environment variables to be
                                set on all ranks.
                                Example: `-x DATASET=/scratch/data TF_CPP_MIN_LOG_LEVEL=1`
                             """,
                      dest = "setenv",
                      type = str,
                      nargs="+",
                      default=[])
  perf_group = parser.add_argument_group('Performance tuning')
  perf_group.add_argument("--fusion-threshold",
                          help="tensor fusion threshold [kilobytes]; use 0 to disable tensor fusion",
                          dest = "fusion_threshold_kb",
                          type = int,
                          default = None)
  perf_group.add_argument("--pin-to-socket",
                          help="pin each rank to a socket using `numactl`, based on rank id",
                          dest = "pin_to_socket",
                          action='store_true',
                          default=False)
  perf_group.add_argument("--pin-memory-to-socket",
                          help="""pin memory allocation for each rank to a socket using `numactl`,
                                  based on rank id [default: False - memory will only be
                                  preferentially allocated from the current socket
                                  (`numactl --preferred`)]
                               """,
                          dest = "pin_mem_to_socket",
                          action='store_true',
                          default=False)
  perf_group.add_argument("--python-interpreter",
                          help="""use a specific Python interpreter instead of the default
                                  `python` found in $PATH.
                                  Pass an empty string to this option to run your SCRIPT without an
                                  intepreter.
                               """,
                          dest = "python_interpreter",
                          type = str,
                          default = "python")
  cleanup_group = parser.add_argument_group('Cleanup')
  cleanup_group.add_argument("--cleanup",
                             help = "clean up remaining processes after an abnormal termination",
                             dest = "cleanup",
                             action = 'store_true',
                             default = False)
  cleanup_group.add_argument("--force",
                             help = "force termination of cleaned up processes",
                             dest = "force",
                             action = 'store_true',
                             default = False)
  parser.add_argument("--dry-run",
                      help = "print generated files and execution command",
                      dest = "dry_run",
                      action='store_true',
                      default = False)
  parser.add_argument("--version",
                      action = 'version',
                      version = generate_version_message())
  parser.add_argument('script', nargs='+', metavar='-- SCRIPT')
  return parser

def parse_args():
  parser = create_parser()
  args = parser.parse_args()

  if args.npernode_single_node:
    if args.npernode:
      sys.exit("[TNT_CLI] Use either `-n`, `--n-per-node`, or `--devices-per-node` argument.")
    else:
      args.npernode = args.npernode_single_node

  if args.force:
    if not args.cleanup:
      sys.exit("[TNT_CLI] Use `--force` only in conjunction with `--cleanup` " \
               "to force termination of Tarantella processes.")
  return args

def generate_version_message():
  msg = ["Tarantella {}".format(tnt_version),
         "Path: {}".format(os.path.dirname(os.path.abspath(__file__))),
         "Copyright (C) 2022 Fraunhofer"]
  return "\n".join(msg)

def tnt_run_message(command_list, hostfile_path, exec_script_path):
  msg = ""
  if not hostfile_path is None:
    msg += "\n{}\nGenerated hostfile:\n".format("="*80)
    with open(hostfile_path, 'r') as f:
      msg += "============= {} =============\n{}\n".format(hostfile_path,
                                                           "".join(f.readlines()))
  if not exec_script_path is None:
    msg += "\n{}\nGenerated script:\n".format("="*80)
    with open(exec_script_path, 'r') as f:
      msg += "============= {} =============\n{}\n".format(exec_script_path,
                                                           "".join(f.readlines()))
  msg += "\n{}".format("="*80)
  msg += "\nCommand:\n\t{}\n".format(" ".join(command_list))
  return msg

def generate_dry_run_message(command_list, hostfile_path, exec_script_path):
  msg = "\n{}".format("="*80)
  msg += "\n{0}{1}DRY RUN {1}{0}\n".format("="*6, " "*30)
  msg += tnt_run_message(command_list, hostfile_path, exec_script_path)
  return msg

def generate_run_error_message(e, hostfile_path = None,
                               executed_script_path = None):
  error_string = ""
  if not e.stdout is None:
    error_string += "============= STDOUT =============\n{}\n".format(e.stdout)
  if not e.stderr is None:
    error_string += "============= STDERR =============\n{}\n".format(e.stderr)
  error_string += tnt_run_message(e.cmd, hostfile_path = hostfile_path,
                                  exec_script_path = executed_script_path)
  error_string += "[TNT_CLI] Execution failed with status {}".format(e.returncode)
  return error_string

def get_numa_nodes_count():
  out = subprocess.run(['numactl', '--hardware'], stdout=subprocess.PIPE)
  out = out.stdout.decode("utf-8")
  out = out.split('\n')[0].split()[1]
  return int(out) if out.isdigit() else 0

def get_numa_prefix(npernode):
  node_count = get_numa_nodes_count()
  if node_count == 0 or npernode == 0 or node_count < npernode:
    raise ValueError(f"[TNT_CLI] Cannot pin {npernode} ranks to {node_count} NUMA nodes. " \
                     f"`-n`, `--n-per-node`, or `--devices-per-node` value should be less" \
                     f" than or equal to {node_count} (available NUMA nodes).")
  else:
    if node_count != npernode:
      logger.warn(f"Pinning {npernode} ranks to NUMA nodes on each host " \
                  f"(available NUMA nodes: {node_count}).")
    command = f"socket=$(( $GASPI_RANK % {npernode} ))\n"
    command += f"numactl --cpunodebind=$socket --membind=$socket"
  return command

class TarantellaCLI:
  def __init__(self, hostlist, num_gpus_per_node, num_cpus_per_node, args):
    self.args = args

    self.command_list = args.script
    self.num_gpus_per_node = num_gpus_per_node
    # compute number of ranks per node to create the hostfile
    self.npernode = num_gpus_per_node
    device_type = "GPUs"
    if self.npernode == 0:
      self.npernode = num_cpus_per_node
      device_type = "CPU processes"

    self.hostlist = hostlist
    self.nranks = len(hostlist) * self.npernode
    self.hostfile = file_man.HostFile(hostlist, self.npernode)
    self.executable_script = self.generate_executable_script()

    logger.info("Starting Tarantella on {} devices ({} nodes x {} {})".format(self.nranks,
                len(hostlist), self.npernode, device_type))

  def generate_interpreter(self):
    interpreter = "python"

    if self.args.pin_to_socket:
      path_to_numa = shutil.which("numactl")
      if path_to_numa is None:
        raise FileNotFoundError("[TNT_CLI] Cannot execute `numactl` as required by " \
                                "the `--pin-to-socket` flag; make sure that `numactl` " \
                                "is installed and has been added to the current `PATH`.")
      interpreter = f"{get_numa_prefix(self.npernode)} {interpreter}"

    return interpreter

  def get_absolute_path(self, module):
    return module.__file__

  def generate_executable_script(self):
    header = "#!/bin/bash\n"
    header += "cd {}".format(os.path.abspath(os.getcwd()))

    environment = env_config.gen_exports_from_dict(env_config.collect_environment_variables()) + \
                  env_config.gen_exports_from_dict(env_config.collect_tensorflow_variables()) + \
                  env_config.gen_exports_from_dict(env_config.collect_tarantella_variables()) + \
                  env_config.gen_exports_from_dict(env_config.get_tnt_variables_from_args(self.args)) + \
                  env_config.gen_exports_from_dict(env_config.get_tnt_gpus(self.num_gpus_per_node)) + \
                  env_config.gen_exports_from_dict(env_config.get_environment_vars_from_args(self.args))

    command = f"{self.generate_interpreter()} {' '.join(self.command_list)}"
    return file_man.GPIScriptFile(header, environment, command, dir = os.getcwd())
  
  def generate_cleanup_script(self):
    header = "#!/bin/bash\n"
    environment = env_config.gen_exports_from_dict(env_config.collect_environment_variables())

    command = f"{self.generate_interpreter()} {self.get_absolute_path(tarantella_cleanup)} " \
                                              f"--proc_names {self.command_list[0]} " \
                                              f"--skip_gaspi_pid {os.getpid()} " \
                                              "--gaspi_rank $GASPI_RANK"
    if self.args.force:
      command = command + " --force"
    return file_man.GPIScriptFile(header, environment, command, dir = os.getcwd())

  def run(self):
    if self.args.cleanup:
      self.clean_up_run()
    else:
      self.normal_run()

  def normal_run(self):
    self.execute_with_gaspi_run(self.nranks, self.hostfile, self.executable_script,
                              self.args.dry_run)

  def clean_up_run(self):
    cleanup_script = self.generate_cleanup_script()
    hostfile = file_man.HostFile(self.hostlist, 1)
    self.execute_with_gaspi_run(len(self.hostlist), hostfile, cleanup_script, self.args.dry_run)
    logger.debug(f'Cleanup executed on {len(self.hostlist)} nodes.')

  def execute_with_gaspi_run(self, nranks, hostfile, executable_script, dry_run = False):
    with hostfile, executable_script:
      command_list = [env_config.path_to_gaspi_run(),
                      "-n", str(nranks),
                      "-m", hostfile.name,
                      executable_script.filename]
      if dry_run:
        print(generate_dry_run_message(command_list, hostfile.name,
                                       executable_script.filename))
        return

      try:
        subprocess.run(command_list,
                       check = True,
                       cwd = os.getcwd(),
                       stdout = None, stderr = None)
      except KeyboardInterrupt:
        logger.info('Execution interrupted: running cleanup')
        self.clean_up_run()
      except subprocess.CalledProcessError as e:
        sys.exit(generate_run_error_message(e, hostfile.name, executable_script.filename))
