#!/bin/bash
source ~/venvOPnew/bin/activate
export PYTHONPATH="$PWD":$PYTHONPATH

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
#cd ./tools/sim/ && ./bridge_for_scenario_runner.py --cruise_lead $1 --init_dist $2 --cruise_lead2 $3 --low_quality
cd ./tools/sim/
python bridge_to_get_lane_bound_new.py
cd $DIR
