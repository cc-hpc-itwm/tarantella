import sys
import os
import signal
import argparse
import subprocess
try:
  import psutil
except:
  sys.exit(f"psutil package should be installed inorder to use --clean-up option. Clean up might not work for this run.")

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
    result = 999999
    print(f"Couldn't kill processes : {e}")
  return result

def get_pid_by_name_psutil(process_name):
  """
  uses psutil to get pid. Pure python way
  """
  pids = []
  # iterate over the all the running process
  for proc in psutil.process_iter():
    try:
      pinfo = proc.as_dict(attrs=['pid', 'name', 'username'])
      # check if process name contains the given name string.
      if process_name.lower() in pinfo['name'].lower() and pinfo['username'] == 'kadur' :
        pids.append(pinfo['pid'])
    except (psutil.NoSuchProcess, psutil.AccessDenied) :
      print("process not found/Access is denied")
      pass
  return pids

def kill_processes(proc_names):
  for proc_name in proc_names:
    pids = get_pid_by_name(proc_name)
    for pid in pids:
      print(f"Terminating process with pid : {pid}")
      os.kill(int(pid), signal.SIGKILL)

if __name__ == '__main__':
  args = parser.parse_args()
  kill_processes(args.proc_names)
 