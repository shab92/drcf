#!/usr/bin/env bash
# Usage: ./run.sh build/test_distaead [args]   starts B1 (party 1) and B2 (party 2) locally
set -e
BIN=$1; shift
"$BIN" 1 "$@" & P1=$!
sleep 0.3
"$BIN" 2 "$@"
wait $P1
