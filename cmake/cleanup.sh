#!/usr/bin/env bash

procs=`ps aux | grep --regexp="\(py\)\?test" | grep -v ctest | grep -v grep`
if [ -n "$procs" ] ;then
  ps aux | grep --regexp="\(py\)\?test" | grep -v ctest | grep -v grep | awk '{print $2}' | xargs kill 2>&1 > /dev/null
fi
