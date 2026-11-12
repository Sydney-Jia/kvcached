#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

# Run the dependency-isolated tests classified for hosted CPU CI.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON="${PYTHON:-python}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

tests=()
# A failed classifier inside process substitution does not fail the parent
# shell. Capture it first so an empty list cannot launch device tests as well.
test_paths="$("${PYTHON}" tools/check_test_classification.py --list-category cpu)"
if [[ -z "${test_paths}" ]]; then
  echo "CPU test classification returned no tests" >&2
  exit 1
fi
while IFS= read -r test_path; do
  tests+=("${test_path}")
done <<< "${test_paths}"

"${PYTHON}" -m pytest "${tests[@]}" "$@"
