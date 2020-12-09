#!/usr/bin/env python
import argparse
import logging
import os
import shutil
import subprocess
import sys
import shlex

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


TNT_PRIVATE_REPO = "git@gitlab.itwm.fraunhofer.de:carpenamarie/hpdlf.git"
TNT_PRIVATE_NAME = "hpdlf"

TNT_ARCHIVE_NAME = "tarantella"

TNT_GITHUB_REPO = "https://github.com/cc-hpc-itwm/tarantella"
TNT_GITHUB_NAME = "tarantella"

def parse_args():
  parser = argparse.ArgumentParser()

  parser.add_argument("--version",
                      default = None,
                      help = "Tarantella version to export to the public repo",
                      dest = "version")
  parser.add_argument("--message",
                      default = None,
                      help = "Commit message on the public repo",
                      dest = "message")
  args = parser.parse_args()
  return args


def run_command_list(command_list, root_dir):
  commands = "\n".join(command_list)
  logger.info("Executing commands:\n{0}\n{1}\n{0}".format("="*60, commands))
  try:
    result = subprocess.run(commands,
                check = True,
                cwd = root_dir,
                shell = True,
                stdout = None, stderr = None,)
  except subprocess.CalledProcessError as e:
    sys.exit("Error running commands: \n{}".format(command_list))


def git_clone_and_go_to_branch(repo_url, repo_name, version = None):
  command_list = ["git clone {}".format(repo_url),
                  "cd {}".format(repo_name)
                  ]
  if version is None:
    command_list += ["git checkout master"]
  else:
    command_list += ["git fetch",
                     "git checkout v{}".format(version)]
  return command_list

def run(version, commit_message):
  root_dir = os.getcwd()

  source_package_name = "{}-{}-src".format(TNT_ARCHIVE_NAME, version)
  source_package_path = os.path.join(os.path.join(root_dir, TNT_PRIVATE_NAME), "build")
  source_package_path = os.path.join(source_package_path, source_package_name)

  logger.info("Cloning local repo and creating package")
  command_list = git_clone_and_go_to_branch(TNT_PRIVATE_REPO, TNT_PRIVATE_NAME, version)
  command_list += [ "mkdir -p build",
                    "cd build",
                    "cmake ../",
                    "make package_source",
                    "tar -xvf {}.tar.gz".format(source_package_name)
                    ]
  run_command_list(command_list, root_dir)

  if commit_message is None:
    commit_message = "v{}".format(version)
  logger.info("Cloning github repo and adding commit with message \"{}\"".format(commit_message))
  github_commands = git_clone_and_go_to_branch(TNT_GITHUB_REPO, TNT_GITHUB_NAME)
  github_commands += ["cp -r {}/* ./".format(source_package_path),
                      "git add --all",
                      "git commit -m \"{}\"".format(commit_message)
                      ]
  run_command_list(github_commands, root_dir)


if __name__ == "__main__":
  args = parse_args()
  logger.setLevel(logging.INFO)
  run(args.version, args.message)