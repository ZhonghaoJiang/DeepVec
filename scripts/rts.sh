
export CUDA_VISIBLE_DEVICES="0"

python3 rq-rts-2.py -dl_model "./RNNModels/mnist_demo/models/mnist_lstm.h5" -model_type "lstm" -dataset "mnist" -mode "ori" 
python3 rq-rts-2.py -dl_model "./RNNModels/mnist_demo/models/mnist_blstm.h5" -model_type "blstm" -dataset "mnist" -mode "ori" 
python3 rq-rts-2.py -dl_model "./RNNModels/mnist_demo/models/mnist_gru.h5" -model_type "gru" -dataset "mnist" -mode "ori" 
python3 rq-rts-2.py -dl_model "./RNNModels/mnist_demo/models/mnist_lstm.h5" -model_type "lstm" -dataset "mnist" -mode "aug" 
python3 rq-rts-2.py -dl_model "./RNNModels/mnist_demo/models/mnist_blstm.h5" -model_type "blstm" -dataset "mnist" -mode "aug" 
python3 rq-rts-2.py -dl_model "./RNNModels/mnist_demo/models/mnist_gru.h5" -model_type "gru" -dataset "mnist" -mode "aug" 


python3 rq-rts-2.py -dl_model "./RNNModels/fashion_demo/models/fashion_lstm.h5" -model_type "lstm" -dataset "fashion" -mode "ori" 
python3 rq-rts-2.py -dl_model "./RNNModels/fashion_demo/models/fashion_blstm.h5" -model_type "blstm" -dataset "fashion" -mode "ori" 
python3 rq-rts-2.py -dl_model "./RNNModels/fashion_demo/models/fashion_gru.h5" -model_type "gru" -dataset "fashion" -mode "ori" 
python3 rq-rts-2.py -dl_model "./RNNModels/fashion_demo/models/fashion_lstm.h5" -model_type "lstm" -dataset "fashion" -mode "aug" 
python3 rq-rts-2.py -dl_model "./RNNModels/fashion_demo/models/fashion_blstm.h5" -model_type "blstm" -dataset "fashion" -mode "aug" 
python3 rq-rts-2.py -dl_model "./RNNModels/fashion_demo/models/fashion_gru.h5" -model_type "gru" -dataset "fashion" -mode "aug" 


python3 rq-rts-2.py -dl_model "./RNNModels/svhn_demo/models/svhn_lstm.h5" -model_type "lstm" -dataset "svhn" -mode "ori" 
python3 rq-rts-2.py -dl_model "./RNNModels/svhn_demo/models/svhn_blstm.h5" -model_type "blstm" -dataset "svhn" -mode "ori" 
python3 rq-rts-2.py -dl_model "./RNNModels/svhn_demo/models/svhn_gru.h5" -model_type "gru" -dataset "svhn" -mode "ori" 
python3 rq-rts-2.py -dl_model "./RNNModels/svhn_demo/models/svhn_lstm.h5" -model_type "lstm" -dataset "svhn"  -mode "aug"
python3 rq-rts-2.py -dl_model "./RNNModels/svhn_demo/models/svhn_blstm.h5" -model_type "blstm" -dataset "svhn"  -mode "aug"
python3 rq-rts-2.py -dl_model "./RNNModels/svhn_demo/models/svhn_gru.h5" -model_type "gru" -dataset "svhn"  -mode "aug"



python3 rq-rts-1.py -dl_model "./RNNModels/mnist_demo/models/mnist_lstm.h5" -model_type "lstm" -dataset "mnist" 
python3 rq-rts-1.py -dl_model "./RNNModels/mnist_demo/models/mnist_blstm.h5" -model_type "blstm" -dataset "mnist" 
python3 rq-rts-1.py -dl_model "./RNNModels/mnist_demo/models/mnist_gru.h5" -model_type "gru" -dataset "mnist" 

python3 rq-rts-1.py -dl_model "./RNNModels/fashion_demo/models/fashion_lstm.h5" -model_type "lstm" -dataset "fashion" 
python3 rq-rts-1.py -dl_model "./RNNModels/fashion_demo/models/fashion_blstm.h5" -model_type "blstm" -dataset "fashion" 
python3 rq-rts-1.py -dl_model "./RNNModels/fashion_demo/models/fashion_gru.h5" -model_type "gru" -dataset "fashion" 


python3 rq-rts-1.py -dl_model "./RNNModels/svhn_demo/models/svhn_lstm.h5" -model_type "lstm" -dataset "svhn" 
python3 rq-rts-1.py -dl_model "./RNNModels/svhn_demo/models/svhn_blstm.h5" -model_type "blstm" -dataset "svhn" 
python3 rq-rts-1.py -dl_model "./RNNModels/svhn_demo/models/svhn_gru.h5" -model_type "gru" -dataset "svhn" 



python3 rq-rts-3.py -dl_model "./RNNModels/mnist_demo/models/mnist_lstm_ori.h5" -model_type "lstm" -dataset "mnist"
python3 rq-rts-3.py -dl_model "./RNNModels/mnist_demo/models/mnist_blstm_ori.h5" -model_type "blstm" -dataset "mnist"
python3 rq-rts-3.py -dl_model "./RNNModels/mnist_demo/models/mnist_gru_ori.h5" -model_type "gru" -dataset "mnist"

python3 rq-rts-3.py -dl_model "./RNNModels/fashion_demo/models/fashion_lstm_ori.h5" -model_type "lstm" -dataset "fashion"
python3 rq-rts-3.py -dl_model "./RNNModels/fashion_demo/models/fashion_blstm_ori.h5" -model_type "blstm" -dataset "fashion"
python3 rq-rts-3.py -dl_model "./RNNModels/fashion_demo/models/fashion_gru_ori.h5" -model_type "gru" -dataset "fashion"

python3 rq-rts-3.py -dl_model "./RNNModels/svhn_demo/models/svhn_lstm_ori.h5" -model_type "lstm" -dataset "svhn"
python3 rq-rts-3.py -dl_model "./RNNModels/svhn_demo/models/svhn_blstm_ori.h5" -model_type "blstm" -dataset "svhn"
python3 rq-rts-3.py -dl_model "./RNNModels/svhn_demo/models/svhn_gru_ori.h5" -model_type "gru" -dataset "svhn"