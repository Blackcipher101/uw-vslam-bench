#!/bin/bash
# Full laptop runs (sequential so the timings do not disturb each other). Logs flush per 100 frames.
cd "$(dirname "$(readlink -f "$0")")"
B=../build/run_bench.sh
$B --gpu-keepalive --track-out ntnu06_gpu_klt.uwtrk --report-every 100 2>&1 | tee closed_keepalive.log
$B --gpu-keepalive --open-loop --no-cpu-stages --report-every 100 2>&1 | tee open_loop_keepalive.log
$B --max-frames 1500 --no-cpu-stages --report-every 100 2>&1 | tee closed_default_clocks.log
