#!/usr/bin/env bash
# Block until every given path exists (used to chain jobs on data produced by another machine).
for path in "$@"; do
  until [[ -e "${path}" ]]; do sleep 120; done
done
