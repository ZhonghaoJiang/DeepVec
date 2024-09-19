python3 rq4_100tests.py -dl_model "./RNNModels/mnist_demo/models/mnist_lstm_ori.h5" -model_type "lstm" -dataset "mnist" &
python3 rq4_100tests.py -dl_model "./RNNModels/mnist_demo/models/mnist_blstm_ori.h5" -model_type "blstm" -dataset "mnist" &
python3 rq4_100tests.py -dl_model "./RNNModels/mnist_demo/models/mnist_gru_ori.h5" -model_type "gru" -dataset "mnist" &

python3 rq4_100tests.py -dl_model "./RNNModels/fashion_demo/models/fashion_lstm_ori.h5" -model_type "lstm" -dataset "fashion" &
python3 rq4_100tests.py -dl_model "./RNNModels/fashion_demo/models/fashion_blstm_ori.h5" -model_type "blstm" -dataset "fashion" &
python3 rq4_100tests.py -dl_model "./RNNModels/fashion_demo/models/fashion_gru_ori.h5" -model_type "gru" -dataset "fashion" &


python3 rq4_100tests.py -dl_model "./RNNModels/agnews_demo/models/agnews_lstm_ori.h5" -model_type "lstm" -dataset "agnews" &
python3 rq4_100tests.py -dl_model "./RNNModels/agnews_demo/models/agnews_blstm_ori.h5" -model_type "blstm" -dataset "agnews" &
python3 rq4_100tests.py -dl_model "./RNNModels/agnews_demo/models/agnews_gru_ori.h5" -model_type "gru" -dataset "agnews" &


python3 rq4_100tests.py -dl_model "./RNNModels/snips_demo/models/snips_lstm_ori.h5" -model_type "lstm" -dataset "snips" &
python3 rq4_100tests.py -dl_model "./RNNModels/snips_demo/models/snips_blstm_ori.h5" -model_type "blstm" -dataset "snips" &
python3 rq4_100tests.py -dl_model "./RNNModels/snips_demo/models/snips_gru_ori.h5" -model_type "gru" -dataset "snips" &


python3 rq4_100tests.py -dl_model "./RNNModels/svhn_demo/models/svhn_lstm_ori.h5" -model_type "lstm" -dataset "svhn" &
python3 rq4_100tests.py -dl_model "./RNNModels/svhn_demo/models/svhn_blstm_ori.h5" -model_type "blstm" -dataset "svhn" &
python3 rq4_100tests.py -dl_model "./RNNModels/svhn_demo/models/svhn_gru_ori.h5" -model_type "gru" -dataset "svhn" &

wait