#!/usr/bin/env bash

procs=`ps aux | grep --regexp="\(py\)\?[Tt]est" | grep -v cleanup.sh | grep -v ctest | grep -v grep  | grep -v run_`
if [ -n "$procs" ] ;then
  ps aux | grep --regexp="\(py\)\?[Tt]est" | grep -v cleanup.sh | grep -v ctest | grep -v grep | grep -v run_ | awk '{print $2}' | xargs kill || 1 2>&1 > /dev/null
fi

exit 0
