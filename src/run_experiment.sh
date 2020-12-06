#!/usr/bin/env bash
./run.sh 10 ./main.py -np -sr learning_rate_experiment -E 100 -en L_RATE:0.007 -LR 0.0007;
./run.sh 10 ./main.py -np -sr learning_rate_experiment -E 100 -en L_RATE:0.008 -LR 0.0008;
./run.sh 10 ./main.py -np -sr learning_rate_experiment -E 100 -en L_RATE:0.009 -LR 0.0009;
./run.sh 10 ./main.py -np -sr learning_rate_experiment -E 100 -en L_RATE:0.01 -LR 0.001;
./run.sh 10 ./main.py -np -sr learning_rate_experiment -E 100 -en L_RATE:0.02 -LR 0.002;
./run.sh 10 ./main.py -np -sr learning_rate_experiment -E 100 -en L_RATE:0.03 -LR 0.003;
./run.sh 10 ./main.py -np -sr learning_rate_experiment -E 100 -en L_RATE:0.04 -LR 0.004;
