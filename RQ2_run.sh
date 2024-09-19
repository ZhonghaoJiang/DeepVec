#!/bin/bash

python3 rq2.py -dl_model "./RNNModels/mnist_demo/models/mnist_lstm.h5" -model_type "lstm" -dataset "mnist" &
python3 rq2.py -dl_model "./RNNModels/mnist_demo/models/mnist_blstm.h5" -model_type "blstm" -dataset "mnist" &
python3 rq2.py -dl_model "./RNNModels/mnist_demo/models/mnist_gru.h5" -model_type "gru" -dataset "mnist" &

python3 rq2.py -dl_model "./RNNModels/fashion_demo/models/fashion_lstm.h5" -model_type "lstm" -dataset "fashion" &
python3 rq2.py -dl_model "./RNNModels/fashion_demo/models/fashion_blstm.h5" -model_type "blstm" -dataset "fashion" &
python3 rq2.py -dl_model "./RNNModels/fashion_demo/models/fashion_gru.h5" -model_type "gru" -dataset "fashion" &

python3 rq2.py -dl_model "./RNNModels/agnews_demo/models/agnews_lstm.h5" -model_type "lstm" -dataset "agnews" &
python3 rq2.py -dl_model "./RNNModels/agnews_demo/models/agnews_blstm.h5" -model_type "blstm" -dataset "agnews" &
python3 rq2.py -dl_model "./RNNModels/agnews_demo/models/agnews_gru.h5" -model_type "gru" -dataset "agnews" &

python3 rq2.py -dl_model "./RNNModels/snips_demo/models/snips_lstm.h5" -model_type "lstm" -dataset "snips" &
python3 rq2.py -dl_model "./RNNModels/snips_demo/models/snips_blstm.h5" -model_type "blstm" -dataset "snips" &
python3 rq2.py -dl_model "./RNNModels/snips_demo/models/snips_gru.h5" -model_type "gru" -dataset "snips" &

python3 rq2.py -dl_model "./RNNModels/svhn_demo/models/svhn_lstm.h5" -model_type "lstm" -dataset "svhn" &
python3 rq2.py -dl_model "./RNNModels/svhn_demo/models/svhn_blstm.h5" -model_type "blstm" -dataset "svhn" &
python3 rq2.py -dl_model "./RNNModels/svhn_demo/models/svhn_gru.h5" -model_type "gru" -dataset "svhn" &

wait