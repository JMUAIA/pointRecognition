#!/bin/bash
LOG=./log/train-origin-`date +%Y-%m-%d-%H-%M-%S`.log
~/software/caffe/build/tools/caffe train -solver solver.prototxt -model resnet_train_val.prototxt -gpu 0 2>&1 | tee $LOG
