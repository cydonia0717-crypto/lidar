#!/usr/bin/env bash
set -eo pipefail
export DEBIAN_FRONTEND=noninteractive
STAGE=${1:-c}
case "$STAGE" in c|d) ;; *) exit 2 ;; esac
OUT=/workspace/results/dra-mppi-stage-$STAGE
mkdir -p "$OUT"
apt-get update
apt-get install -y --no-install-recommends build-essential git python3-pip python3-dev python3-yaml libeigen3-dev clang libc++-dev libc++abi-dev \
  ros-noetic-tf ros-noetic-nav-msgs ros-noetic-visualization-msgs ros-noetic-message-generation ros-noetic-derived-object-msgs ros-noetic-message-filters
git config --global --add safe.directory /workspace/_upstream/dra-mppi
test "$(git -C /workspace/_upstream/dra-mppi rev-parse HEAD)" = 6d76ab2878c503121544b992c0e5706c6c6dece0
python3 -m pip install 'pip==24.0'
python3 -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install --ignore-installed -r /workspace/research/dra_mppi/requirements-cpu.txt
CC=clang CXX=clang++ python3 -m pip install ghalton==0.6.1
python3 -m pip freeze > "$OUT/pip-freeze.txt"
dpkg-query -W > "$OUT/dpkg-packages.txt"
python3 /workspace/research/dra_mppi/prepare_ros_workspace.py --upstream /workspace/_upstream/dra-mppi --workspace /tmp/dra_ros_ws --harness /workspace/research/dra_mppi --output "$OUT"
source /opt/ros/noetic/setup.bash
cd /tmp/dra_ros_ws
catkin_make -j2 -DCMAKE_BUILD_TYPE=Release 2>&1 | tee "$OUT/build.log"
source devel/setup.bash
roscore > "$OUT/roscore.log" 2>&1 &
ROSCORE_PID=$!
trap 'kill "$ROSCORE_PID" 2>/dev/null || true' EXIT
sleep 3
if [ "$STAGE" = d ]; then
  python3 /workspace/research/dra_mppi/run_ros_social.py --upstream /workspace/_upstream/dra-mppi --output "$OUT" --protocol /workspace/research/dra_mppi/protocol-stage-d.json 2>&1 | tee "$OUT/social.log"
else
  python3 /workspace/research/dra_mppi/run_ros_social.py --upstream /workspace/_upstream/dra-mppi --output "$OUT" 2>&1 | tee "$OUT/social.log"
fi
