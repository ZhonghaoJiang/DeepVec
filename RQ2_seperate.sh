#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
python3 rq2_seperate.py -dl_model "./RNNModels/mnist_demo/models/mnist_lstm.h5" -model_type "lstm" -dataset "mnist" -mode "ori" &
python3 rq2_seperate.py -dl_model "./RNNModels/mnist_demo/models/mnist_blstm.h5" -model_type "blstm" -dataset "mnist" -mode "ori" &
python3 rq2_seperate.py -dl_model "./RNNModels/mnist_demo/models/mnist_gru.h5" -model_type "gru" -dataset "mnist" -mode "ori" &
python3 rq2_seperate.py -dl_model "./RNNModels/mnist_demo/models/mnist_lstm.h5" -model_type "lstm" -dataset "mnist" -mode "aug" &
python3 rq2_seperate.py -dl_model "./RNNModels/mnist_demo/models/mnist_blstm.h5" -model_type "blstm" -dataset "mnist" -mode "aug" &
python3 rq2_seperate.py -dl_model "./RNNModels/mnist_demo/models/mnist_gru.h5" -model_type "gru" -dataset "mnist" -mode "aug" &


python3 rq2_seperate.py -dl_model "./RNNModels/fashion_demo/models/fashion_lstm.h5" -model_type "lstm" -dataset "fashion" -mode "ori" &
python3 rq2_seperate.py -dl_model "./RNNModels/fashion_demo/models/fashion_blstm.h5" -model_type "blstm" -dataset "fashion" -mode "ori" &
python3 rq2_seperate.py -dl_model "./RNNModels/fashion_demo/models/fashion_gru.h5" -model_type "gru" -dataset "fashion" -mode "ori" &
python3 rq2_seperate.py -dl_model "./RNNModels/fashion_demo/models/fashion_lstm.h5" -model_type "lstm" -dataset "fashion" -mode "aug" &
python3 rq2_seperate.py -dl_model "./RNNModels/fashion_demo/models/fashion_blstm.h5" -model_type "blstm" -dataset "fashion" -mode "aug" &
python3 rq2_seperate.py -dl_model "./RNNModels/fashion_demo/models/fashion_gru.h5" -model_type "gru" -dataset "fashion" -mode "aug" &

export CUDA_VISIBLE_DEVICES=1
python3 rq2_seperate.py -dl_model "./RNNModels/agnews_demo/models/agnews_lstm.h5" -model_type "lstm" -dataset "agnews" -mode "ori" &
python3 rq2_seperate.py -dl_model "./RNNModels/agnews_demo/models/agnews_blstm.h5" -model_type "blstm" -dataset "agnews" -mode "ori" &
python3 rq2_seperate.py -dl_model "./RNNModels/agnews_demo/models/agnews_gru.h5" -model_type "gru" -dataset "agnews" -mode "ori" &
python3 rq2_seperate.py -dl_model "./RNNModels/agnews_demo/models/agnews_lstm.h5" -model_type "lstm" -dataset "agnews" -mode "aug" &
python3 rq2_seperate.py -dl_model "./RNNModels/agnews_demo/models/agnews_blstm.h5" -model_type "blstm" -dataset "agnews" -mode "aug" &
python3 rq2_seperate.py -dl_model "./RNNModels/agnews_demo/models/agnews_gru.h5" -model_type "gru" -dataset "agnews" -mode "aug" &
#
export CUDA_VISIBLE_DEVICES=2
python3 rq2_seperate.py -dl_model "./RNNModels/snips_demo/models/snips_lstm.h5" -model_type "lstm" -dataset "snips" -mode "ori" &
python3 rq2_seperate.py -dl_model "./RNNModels/snips_demo/models/snips_blstm.h5" -model_type "blstm" -dataset "snips" -mode "ori" &
python3 rq2_seperate.py -dl_model "./RNNModels/snips_demo/models/snips_gru.h5" -model_type "gru" -dataset "snips" -mode "ori" &
python3 rq2_seperate.py -dl_model "./RNNModels/snips_demo/models/snips_lstm.h5" -model_type "lstm" -dataset "snips" -mode "aug" &
python3 rq2_seperate.py -dl_model "./RNNModels/snips_demo/models/snips_blstm.h5" -model_type "blstm" -dataset "snips" -mode "aug" &
python3 rq2_seperate.py -dl_model "./RNNModels/snips_demo/models/snips_gru.h5" -model_type "gru" -dataset "snips" -mode "aug" &


python3 rq2_seperate.py -dl_model "./RNNModels/svhn_demo/models/svhn_lstm.h5" -model_type "lstm" -dataset "svhn" -mode "ori" &
python3 rq2_seperate.py -dl_model "./RNNModels/svhn_demo/models/svhn_blstm.h5" -model_type "blstm" -dataset "svhn" -mode "ori" &
python3 rq2_seperate.py -dl_model "./RNNModels/svhn_demo/models/svhn_gru.h5" -model_type "gru" -dataset "svhn" -mode "ori" &
python3 rq2_seperate.py -dl_model "./RNNModels/svhn_demo/models/svhn_lstm.h5" -model_type "lstm" -dataset "svhn"  -mode "aug"&
python3 rq2_seperate.py -dl_model "./RNNModels/svhn_demo/models/svhn_blstm.h5" -model_type "blstm" -dataset "svhn"  -mode "aug"&
python3 rq2_seperate.py -dl_model "./RNNModels/svhn_demo/models/svhn_gru.h5" -model_type "gru" -dataset "svhn"  -mode "aug"&

wait