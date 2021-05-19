import sys
import os
import signal
import argparse
import subprocess

parser = argparse.ArgumentParser(description='Tarantell cleanup')
parser.add_argument('--proc_names', required=True, type=str, nargs='+',
                    help='a list of process names that has to be terminated')
parser.add_argument('--skip_gaspi_pid', type=int, nargs='+', default=-1,
                    help='list of pids of the process that should be skipped from termination')
parser.add_argument('--gaspi_rank', type=int, default=-1,
                    help='current gaspi rank')

def is_master_rank():
  return args.gaspi_rank == 0

def get_pid_by_name(process_name):
  '''
  uses linux command pgrep to extract pid
  '''
  try:
    result = subprocess.run(["pgrep", "-f", process_name], stdout=subprocess.PIPE)
    result = result.stdout.decode('utf-8')
    skip_pid = list()
    # skip gaspi process from being killed only on master rank
    if is_master_rank():
      skip_pid = list(args.skip_gaspi_pid) if not isinstance(args.skip_gaspi_pid, list) else args.skip_gaspi_pid
    # skip the current process from terminating
    skip_pid.append(os.getpid())
    result = [int(res) for res in result.split('\n')[:-1] if int(res) not in skip_pid]
  except (subprocess.CalledProcessError) as e:
    sys.exit(f"[TNT_CLI] Error occured while getting process ids of GPI processes") 
  return result

def kill_processes(proc_names):
  '''
  kill running processes by sending SIGTERM signal
  '''
  for proc_name in proc_names:
    pids = get_pid_by_name(proc_name)
    for pid in pids:
      print(f"Terminating process with pid : {pid}")
      os.kill(int(pid), signal.SIGTERM)

if __name__ == '__main__':
  args = parser.parse_args()
  kill_processes(args.proc_names)
