#!/usr/bin/env bash

procs=`ps aux | grep --regexp="\(py\)\?test" | grep -v cleanup.sh | grep -v ctest | grep -v grep`
if [ -n "$procs" ] ;then
  ps aux | grep --regexp="\(py\)\?test" | grep -v cleanup.sh | grep -v ctest | grep -v grep | awk '{print $2}' | xargs kill || 1 2>&1 > /dev/null 
fi
