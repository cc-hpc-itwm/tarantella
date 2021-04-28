import sys
import os
import signal
import argparse
import subprocess

parser = argparse.ArgumentParser(description='Tarantell cleanup')
parser.add_argument('--proc_names', required=True, type=str, nargs='+',
                    help='a list of process names that has to be terminated')

def get_pid_by_name(process_name):
  '''
  uses linux command pgrep to extract pid
  '''
  # gets pid based on process_name string passed
  try:
    result = subprocess.run(["pgrep", "-f", process_name], stdout=subprocess.PIPE)
    result = result.stdout.decode('utf-8')
    result = [int(res) for res in result.split('\n')[:-1]]
  except (subprocess.CalledProcessError) as e:
    sys.exit(f"[TNT_CLI] Error occured while killing processes started by tarantella command") 
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
 
