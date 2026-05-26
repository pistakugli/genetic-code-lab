#!/usr/bin/env bash
set -uo pipefail

if [ "$#" -lt 3 ]; then
  echo "usage: $0 <log-file> <code-file> <command...>" >&2
  exit 2
fi

log_file=$1
code_file=$2
shift 2

rm -f "$log_file" "$code_file"

"$@" >"$log_file" 2>&1
code=$?
printf '%s' "$code" >"$code_file"
exit "$code"
