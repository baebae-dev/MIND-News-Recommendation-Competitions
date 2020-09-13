
######################################################################################################
# mind2020
# Authors: Hyunsik Jeon(jeon185@snu.ac.kr), SeungCheol Park(ant6si@snu.ac.kr),
#          Yuna Bae(yunabae482@gmail.com), U kang(ukang@snu.ac.kr)
# File: TANR/run.sh
# - run model TANR
#
# Version: 1.0
#######################################################################################################

#!/bin/bash

for i in {0..9}; do
    rm -rf checkpoint/
    python3 src/train.py
    python3 src/evaluate.py
    python3 src/evaluate_file.py
done
