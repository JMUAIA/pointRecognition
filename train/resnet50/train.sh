#!/bin/bash
LOG=modelsresnet50/log/train-origin-`date +%Y-%m-%d-%H-%M-%S`.log
caffe/build/tools/caffe train -solver models/resnet50/solver.prototxt -model  models/resnet50/train_val.prototxt -weights  models/resnet50/ResNet-50-model.caffemodel  -gpu 0,1 2>&1 | tee $LOG
