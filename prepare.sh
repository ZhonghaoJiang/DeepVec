#!/bin/bash

cd RNNModels/mnist_demo
python3 mnist_lstm.py -type "train"
python3 mnist_blstm.py -type "train"
python3 mnist_gru.py -type "train"

cd ../fashion_demo
python3 fashion_lstm.py -type "train"
python3 fashion_blstm.py -type "train"
python3 fashion_gru.py -type "train"

cd ../snips_demo
python3 snips_gru.py -type "train"
python3 snips_lstm.py -type "train"
python3 snips_blstm.py -type "train"


cd ../agnews_demo
python3 agnews_blstm.py -type "train"
python3 agnews_lstm.py -type "train"
python3 agnews_gru.py -type "train"

cd ../../gen_data/gen_test_dataset
python3 dau_mnist.py
python3 dau_fashion.py
python3 dau_agnews.py
python3 dau_snips.py
python3 gen_toselect_dataset.py -dataset "mnist"   # for RQ1 & RQ2
python3 gen_toselect_dataset.py -dataset "fashion"
python3 gen_toselect_dataset.py -dataset "agnews"
python3 gen_toselect_dataset.py -dataset "snips"
python3 gen_toselect_dataset.py -dataset "svhn"

cd ../gen_retrain_dataset
python3 dau_mnist.py
python3 dau_fashion.py
python3 dau_agnews.py
python3 dau_snips.py
python3 gen_retrain.py -dataset "mnist"   # for RQ3
python3 gen_retrain.py -dataset "fashion"   # for RQ3
python3 gen_retrain.py -dataset "agnews"   # for RQ3
python3 gen_retrain.py -dataset "snips"   # for RQ3

cd ../..
python3 ./abstraction_runner.py -test_obj "mnist_lstm"
python3 ./abstraction_runner.py -test_obj "mnist_blstm"
python3 ./abstraction_runner.py -test_obj "mnist_gru"
python3 ./abstraction_runner.py -test_obj "fashion_lstm"
python3 ./abstraction_runner.py -test_obj "fashion_blstm"
python3 ./abstraction_runner.py -test_obj "fashion_gru"
python3 ./abstraction_runner.py -test_obj "agnews_lstm"
python3 ./abstraction_runner.py -test_obj "agnews_blstm"
python3 ./abstraction_runner.py -test_obj "agnews_gru"
python3 ./abstraction_runner.py -test_obj "snips_lstm"
python3 ./abstraction_runner.py -test_obj "snips_blstm"
python3 ./abstraction_runner.py -test_obj "snips_gru"
