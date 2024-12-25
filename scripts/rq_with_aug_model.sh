cd RNNModels/mnist_demo

python3 mnist_lstm.py -type "dau_train" &
python3 mnist_blstm.py -type "dau_train" &
python3 mnist_gru.py -type "dau_train" &

cd ../fashion_demo
python3 fashion_lstm.py -type "dau_train" &
python3 fashion_blstm.py -type "dau_train" &
python3 fashion_gru.py -type "dau_train" &

cd ../agnews_demo
python3 agnews_lstm.py -type "dau_train"
python3 agnews_blstm.py -type "dau_train"
python3 agnews_gru.py -type "dau_train"

cd ../snips_demo
python3 snips_lstm.py -type "dau_train"
python3 snips_blstm.py -type "dau_train"
python3 snips_gru.py -type "dau_train"

cd ../svhn_demo
python3 svhn_lstm.py -type "dau_train" &
python3 svhn_blstm.py -type "dau_train" &
python3 svhn_gru.py -type "dau_train" &

wait
cd ../..


# 定义命令数组，每个命令作为数组的一个元素
commands=(
    "python3 rq4+.py -dl_model ./RNNModels/agnews_demo/models/agnews_lstm_aug.h5 -model_type lstm -dataset agnews"
    "python3 rq4+.py -dl_model ./RNNModels/agnews_demo/models/agnews_blstm_aug.h5 -model_type blstm -dataset agnews"
    "python3 rq4+.py -dl_model ./RNNModels/agnews_demo/models/agnews_gru_aug.h5 -model_type gru -dataset agnews"

     "python3 rq4+.py -dl_model ./RNNModels/snips_demo/models/snips_lstm_aug.h5 -model_type lstm -dataset snips "
     "python3 rq4+.py -dl_model ./RNNModels/snips_demo/models/snips_blstm_aug.h5 -model_type blstm -dataset snips "
     "python3 rq4+.py -dl_model ./RNNModels/snips_demo/models/snips_gru_aug.h5 -model_type gru -dataset snips "

     "python3 rq4+.py -dl_model ./RNNModels/mnist_demo/models/mnist_lstm_dau.h5 -model_type lstm -dataset mnist "
     "python3 rq4+.py -dl_model ./RNNModels/mnist_demo/models/mnist_blstm_dau.h5 -model_type blstm -dataset mnist "
     "python3 rq4+.py -dl_model ./RNNModels/mnist_demo/models/mnist_gru_dau.h5 -model_type gru -dataset mnist "

     "python3 rq4+.py -dl_model ./RNNModels/fashion_demo/models/fashion_lstm_dau.h5 -model_type lstm -dataset fashion "
     "python3 rq4+.py -dl_model ./RNNModels/fashion_demo/models/fashion_blstm_dau.h5 -model_type blstm -dataset fashion "
     "python3 rq4+.py -dl_model ./RNNModels/fashion_demo/models/fashion_gru_dau.h5 -model_type gru -dataset fashion "

     "python3 rq4+.py -dl_model ./RNNModels/svhn_demo/models/svhn_lstm_dau.h5 -model_type lstm -dataset svhn "
     "python3 rq4+.py -dl_model ./RNNModels/svhn_demo/models/svhn_blstm_dau.h5 -model_type blstm -dataset svhn "
     "python3 rq4+.py -dl_model ./RNNModels/svhn_demo/models/svhn_gru_dau.h5 -model_type gru -dataset svhn "
)


for ((i=0; i<${#commands[@]}; i+=3)); do
    export CUDA_VISIBLE_DEVICES="3"
    ${commands[$((i))]} &
    export CUDA_VISIBLE_DEVICES="2"
    ${commands[$((i + 1))]} &
    export CUDA_VISIBLE_DEVICES="1"
    ${commands[$((i + 2))]}
    wait
done


