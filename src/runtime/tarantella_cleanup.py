import argparse
import os
import signal
import subprocess
import sys

parser = argparse.ArgumentParser(description='Tarantella cleanup')
parser.add_argument('--proc_names', required=True, type=str, nargs='+',
                    help='list of process names to be terminated')
parser.add_argument('--skip_gaspi_pid', type=int, nargs='+', default=[],
                    help='list of process IDs that should be skipped from termination')
parser.add_argument('--gaspi_rank', type=int, default=0,
                    help='current GASPI rank')

def is_master_rank():
  return args.gaspi_rank == 0

def get_pids_by_name(process_name):
  '''
  uses linux command pgrep to extract pid
  '''
  try:
    result = subprocess.run(["pgrep", "-f", process_name], stdout=subprocess.PIPE)
    result = result.stdout.decode('utf-8')
  except (subprocess.CalledProcessError) as e:
    sys.exit(f"[TNT_CLI] Error occured while getting process IDs of Tarantella processes")

  # prevent the current `cleanup` process from being terminated
  skip_pid = [os.getpid()]
  if is_master_rank():
    # skip additional processes (i.e., the main Tarantella process running on the first rank)
    skip_pid += args.skip_gaspi_pid

  pids = [int(pid) for pid in result.split('\n') if pid and int(pid) not in skip_pid]
  return pids

def kill_processes(proc_names):
  for proc_name in proc_names:
    print(f"[TNT_CLI] Terminating processes with name : {proc_name}")
    for pid in get_pids_by_name(proc_name):
      os.kill(pid, signal.SIGTERM)

if __name__ == '__main__':
  args = parser.parse_args()
  kill_processes(args.proc_names)
