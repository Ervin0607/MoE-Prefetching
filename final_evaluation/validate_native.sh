#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p final_evaluation/results/submission_validation
exec > >(tee final_evaluation/results/submission_validation/native_validation.log) 2>&1
date -u
uname -a
cmake --version | head -1
g++ --version | head -1
python3 --version
cmake -S LLMSimulator -B LLMSimulator/build-submission -DCMAKE_BUILD_TYPE=Release
cmake --build LLMSimulator/build-submission -j 4
MOE_NATIVE_BUILD="$PWD/LLMSimulator/build-submission" python3 -m unittest discover -s final_evaluation/tests -v
g++ -std=c++17 -I LLMSimulator/src final_evaluation/tests/native_offload_test.cpp -o LLMSimulator/build-submission/native_offload_test
LLMSimulator/build-submission/native_offload_test LLMSimulator/build-submission/engine-test
echo 'Native integration and standalone engine validation passed.'
