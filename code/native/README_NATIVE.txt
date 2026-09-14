Native planner check -- how to rebuild and re-run everything.

  # 1. build the vendored third-party planner (once)
  cd ../../mapf-throughput-envelope/_probes/ext/lacam3
  cmake -B build . -DCMAKE_BUILD_TYPE=Release && cmake --build build -j 8

  # 2. build the guided driver
  cd ../../../../p3-wardrop-mapf-guidance/code/native
  ENV=../../../mapf-throughput-envelope/_probes/ext
  c++ -std=c++17 -O3 -I"$ENV" -o ext_guided ext_guided.cpp \
      "$ENV/lacam3/build/lacam3/liblacam3.a" -lpthread

  # 3. export this repository's maps, guidance fields and endpoint pools
  cd ../.. && ../.venv/bin/python code/export_native_check.py

  # 4. run and score the hash-locked matrices (benchmark corpus is the manuscript's)
  ../.venv/bin/python code/native/run_native_check.py \
      --proto code/native/native_check_protocol_bench.json
  ../.venv/bin/python code/native/analyze_native_check.py \
      --proto code/native/native_check_protocol_bench.json
  ../.venv/bin/python code/native/check_native_check.py
  ../.venv/bin/python code/native/diag_tau_ablation.py
  ../.venv/bin/python code/native/make_native_macros.py

Both runners verify the protocol SHA-256 before doing anything and refuse to run if it moved.
Retained supersedings: results/native_check/runs{,_bench}_floor/ (floor instead of round
quantiser) and results/native_check/runs/ + summary.json (the superseded procedural exp1 corpus).
