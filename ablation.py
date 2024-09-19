import argparse
import numpy as np
from statics import *
from selection_tools import get_selection_information, get_selection_information_new, ats_selection,ats_selection_rank
import keras
import datetime
import sys
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as K

# Specify that the first GPU is available, if there is no GPU, apply: "-1"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True   # Do not occupy all of the video memory, allocate on demand
sess = tf.compat.v1.Session(config=config)

K.set_session(sess)


# RQ1: Bug Detection Rate on {10%, 20%, 50%} selected test set.
if __name__ == '__main__':
    parse = argparse.ArgumentParser("Calculate the inclusiveness for the selected dataset.")
    parse.add_argument('-dl_model', help='path of dl model', required=True)
    parse.add_argument('-model_type', required=True, choices=['lstm', 'blstm', 'gru'])
    parse.add_argument('-dataset', required=True, choices=['mnist', 'snips', 'fashion', 'agnews', 'svhn'])
    args = parse.parse_args()

    if args.model_type == "lstm" and args.dataset == "mnist":
        time_steps = 28
        w2v_path = ""
        from RNNModels.mnist_demo.mnist_lstm import MnistLSTMClassifier

        lstm_classifier = MnistLSTMClassifier()
        model = lstm_classifier.load_hidden_state_model(args.dl_model)
        dense_classifier = MnistLSTMClassifier()
        dense_model = dense_classifier.reload_dense(args.dl_model)

        file_path = "./gen_data/mnist_toselect/mnist_toselect_0.npz"
        wrapper_path = "./RNNModels/mnist_demo/output/lstm/abst_model/wrapper_lstm_mnist_3_10.pkl"
        total_num = 6000

    elif args.model_type == "blstm" and args.dataset == "mnist":
        time_steps = 28
        w2v_path = ""
        from RNNModels.mnist_demo.mnist_blstm import MnistBLSTMClassifier

        lstm_classifier = MnistBLSTMClassifier()
        model = lstm_classifier.load_hidden_state_model(args.dl_model)
        dense_classifier = MnistBLSTMClassifier()
        dense_model = dense_classifier.reload_dense(args.dl_model)

        file_path = "./gen_data/mnist_toselect/mnist_toselect_0.npz"
        wrapper_path = "./RNNModels/mnist_demo/output/blstm/abst_model/wrapper_blstm_mnist_3_10.pkl"
        total_num = 6000

    elif args.model_type == "gru" and args.dataset == "mnist":
        time_steps = 28
        w2v_path = ""
        from RNNModels.mnist_demo.mnist_gru import MnistGRUClassifier

        lstm_classifier = MnistGRUClassifier()
        model = lstm_classifier.load_hidden_state_model(args.dl_model)
        dense_classifier = MnistGRUClassifier()
        dense_model = dense_classifier.reload_dense(args.dl_model)

        file_path = "./gen_data/mnist_toselect/mnist_toselect_0.npz"
        wrapper_path = "./RNNModels/mnist_demo/output/gru/abst_model/wrapper_gru_mnist_3_10.pkl"
        total_num = 6000

    elif args.model_type == "blstm" and args.dataset == "snips":
        time_steps = 16
        from RNNModels.snips_demo.snips_blstm import SnipsBLSTMClassifier

        lstm_classifier = SnipsBLSTMClassifier()
        lstm_classifier.data_path = "./RNNModels/snips_demo/save/standard_data.npz"
        lstm_classifier.embedding_path = "./RNNModels/snips_demo/save/embedding_matrix.npy"
        model = lstm_classifier.load_hidden_state_model(args.dl_model)
        dense_classifier = SnipsBLSTMClassifier()
        dense_model = dense_classifier.reload_dense(args.dl_model)

        file_path = "./gen_data/snips_toselect/snips_toselect_0.csv"
        wrapper_path = "./RNNModels/snips_demo/output/blstm/abst_model/wrapper_blstm_snips_3_10.pkl"
        w2v_path = "./RNNModels/snips_demo/save/w2v_model"
        total_num = 2000

    elif args.model_type == "gru" and args.dataset == "snips":
        time_steps = 16
        from RNNModels.snips_demo.snips_gru import SnipsGRUClassifier

        lstm_classifier = SnipsGRUClassifier()
        lstm_classifier.data_path = "./RNNModels/snips_demo/save/standard_data.npz"
        lstm_classifier.embedding_path = "./RNNModels/snips_demo/save/embedding_matrix.npy"
        model = lstm_classifier.load_hidden_state_model(args.dl_model)
        dense_classifier = SnipsGRUClassifier()
        dense_model = dense_classifier.reload_dense(args.dl_model)

        file_path = "./gen_data/snips_toselect/snips_toselect_0.csv"
        wrapper_path = "./RNNModels/snips_demo/output/gru/abst_model/wrapper_gru_snips_3_10.pkl"
        w2v_path = "./RNNModels/snips_demo/save/w2v_model"
        total_num = 2000

    elif args.model_type == "lstm" and args.dataset == "snips":
        time_steps = 16
        from RNNModels.snips_demo.snips_lstm import SnipsLSTMClassifier

        lstm_classifier = SnipsLSTMClassifier()
        lstm_classifier.data_path = "./RNNModels/snips_demo/save/standard_data.npz"
        lstm_classifier.embedding_path = "./RNNModels/snips_demo/save/embedding_matrix.npy"
        model = lstm_classifier.load_hidden_state_model(args.dl_model)
        dense_classifier = SnipsLSTMClassifier()
        dense_model = dense_classifier.reload_dense(args.dl_model)

        file_path = "./gen_data/snips_toselect/snips_toselect_0.csv"
        wrapper_path = "./RNNModels/snips_demo/output/lstm/abst_model/wrapper_lstm_snips_3_10.pkl"
        w2v_path = "./RNNModels/snips_demo/save/w2v_model"
        total_num = 2000

    elif args.model_type == "lstm" and args.dataset == "fashion":
        time_steps = 28
        w2v_path = ""
        from RNNModels.fashion_demo.fashion_lstm import FashionLSTMClassifier

        lstm_classifier = FashionLSTMClassifier()
        model = lstm_classifier.load_hidden_state_model(args.dl_model)
        dense_classifier = FashionLSTMClassifier()
        dense_model = dense_classifier.reload_dense(args.dl_model)

        file_path = "./gen_data/fashion_toselect/fashion_toselect_0.npz"
        wrapper_path = "./RNNModels/fashion_demo/output/lstm/abst_model/wrapper_lstm_fashion_3_10.pkl"
        total_num = 6000

    elif args.model_type == "blstm" and args.dataset == "fashion":
        time_steps = 28
        w2v_path = ""
        from RNNModels.fashion_demo.fashion_blstm import FashionBLSTMClassifier

        lstm_classifier = FashionBLSTMClassifier()
        model = lstm_classifier.load_hidden_state_model(args.dl_model)
        dense_classifier = FashionBLSTMClassifier()
        dense_model = dense_classifier.reload_dense(args.dl_model)

        file_path = "./gen_data/fashion_toselect/fashion_toselect_0.npz"
        wrapper_path = "./RNNModels/fashion_demo/output/blstm/abst_model/wrapper_blstm_fashion_3_10.pkl"
        total_num = 6000

    elif args.model_type == "gru" and args.dataset == "fashion":
        time_steps = 28
        w2v_path = ""
        from RNNModels.fashion_demo.fashion_gru import FashionGRUClassifier

        lstm_classifier = FashionGRUClassifier()
        model = lstm_classifier.load_hidden_state_model(args.dl_model)
        dense_classifier = FashionGRUClassifier()
        dense_model = dense_classifier.reload_dense(args.dl_model)

        file_path = "./gen_data/fashion_toselect/fashion_toselect_0.npz"
        wrapper_path = "./RNNModels/fashion_demo/output/gru/abst_model/wrapper_gru_fashion_3_10.pkl"
        total_num = 6000

    elif args.model_type == "lstm" and args.dataset == "svhn":
        time_steps = 32
        w2v_path = ""
        from RNNModels.svhn_demo.svhn_lstm import SvhnLSTMClassifier

        lstm_classifier = SvhnLSTMClassifier()
        model = lstm_classifier.load_hidden_state_model(args.dl_model)
        dense_classifier = SvhnLSTMClassifier()
        dense_model = dense_classifier.reload_dense(args.dl_model)

        file_path = "./gen_data/svhn_toselect/svhn_toselect_0.npz"
        wrapper_path = "./RNNModels/svhn_demo/output/lstm/abst_model/wrapper_lstm_svhn_3_10.pkl"
        total_num = 7000

    elif args.model_type == "blstm" and args.dataset == "svhn":
        time_steps = 32
        w2v_path = ""
        from RNNModels.svhn_demo.svhn_blstm import SvhnBLSTMClassifier

        lstm_classifier = SvhnBLSTMClassifier()
        model = lstm_classifier.load_hidden_state_model(args.dl_model)
        dense_classifier = SvhnBLSTMClassifier()
        dense_model = dense_classifier.reload_dense(args.dl_model)

        file_path = "./gen_data/svhn_toselect/svhn_toselect_0.npz"
        wrapper_path = "./RNNModels/svhn_demo/output/blstm/abst_model/wrapper_blstm_svhn_3_10.pkl"
        total_num = 7000

    elif args.model_type == "gru" and args.dataset == "svhn":
        time_steps = 32
        w2v_path = ""
        from RNNModels.svhn_demo.svhn_gru import SvhnGRUClassifier

        lstm_classifier = SvhnGRUClassifier()
        model = lstm_classifier.load_hidden_state_model(args.dl_model)
        dense_classifier = SvhnGRUClassifier()
        dense_model = dense_classifier.reload_dense(args.dl_model)

        file_path = "./gen_data/svhn_toselect/svhn_toselect_0.npz"
        wrapper_path = "./RNNModels/svhn_demo/output/gru/abst_model/wrapper_gru_svhn_3_10.pkl"
        total_num = 7000


    elif args.model_type == "lstm" and args.dataset == "agnews":
        time_steps = 35
        from RNNModels.agnews_demo.agnews_lstm import AGNewsLSTMClassifier

        lstm_classifier = AGNewsLSTMClassifier()
        lstm_classifier.data_path = "./RNNModels/agnews_demo/save/standard_data.npz"
        lstm_classifier.embedding_path = "./RNNModels/agnews_demo/save/embedding_matrix.npy"
        model = lstm_classifier.load_hidden_state_model(args.dl_model)
        dense_classifier = AGNewsLSTMClassifier()
        dense_model = dense_classifier.reload_dense(args.dl_model)

        w2v_path = "./RNNModels/agnews_demo/save/w2v_model"
        file_path = "./gen_data/agnews_toselect/agnews_toselect_0.csv"
        wrapper_path = "./RNNModels/agnews_demo/output/lstm/abst_model/wrapper_lstm_agnews_3_10.pkl"
        total_num = 4560

    elif args.model_type == "blstm" and args.dataset == "agnews":
        time_steps = 35
        from RNNModels.agnews_demo.agnews_blstm import AgnewsBLSTMClassifier

        lstm_classifier = AgnewsBLSTMClassifier()
        lstm_classifier.data_path = "./RNNModels/agnews_demo/save/standard_data.npz"
        lstm_classifier.embedding_path = "./RNNModels/agnews_demo/save/embedding_matrix.npy"
        model = lstm_classifier.load_hidden_state_model(args.dl_model)
        dense_classifier = AgnewsBLSTMClassifier()
        dense_model = dense_classifier.reload_dense(args.dl_model)

        w2v_path = "./RNNModels/agnews_demo/save/w2v_model"
        file_path = "./gen_data/agnews_toselect/agnews_toselect_0.csv"
        wrapper_path = "./RNNModels/agnews_demo/output/blstm/abst_model/wrapper_blstm_agnews_3_10.pkl"
        total_num = 4560

    elif args.model_type == "gru" and args.dataset == "agnews":
        time_steps = 35
        from RNNModels.agnews_demo.agnews_gru import AgnewsGRUClassifier

        lstm_classifier = AgnewsGRUClassifier()
        lstm_classifier.data_path = "./RNNModels/agnews_demo/save/standard_data.npz"
        lstm_classifier.embedding_path = "./RNNModels/agnews_demo/save/embedding_matrix.npy"
        model = lstm_classifier.load_hidden_state_model(args.dl_model)
        dense_classifier = AgnewsGRUClassifier()
        dense_model = dense_classifier.reload_dense(args.dl_model)

        w2v_path = "./RNNModels/agnews_demo/save/w2v_model"
        file_path = "./gen_data/agnews_toselect/agnews_toselect_0.csv"
        wrapper_path = "./RNNModels/agnews_demo/output/gru/abst_model/wrapper_gru_agnews_3_10.pkl"
        total_num = 4560

    else:
        print("The model and data set are incorrect.")
        sys.exit(1)

    vec,vec_woun,vec_wosim,vec_inclu, vec_woun_inclu, vec_wosim_inclu = {}, {}, {},{},{},{}
    pre_li = [10, 20]
    # pre_li = [100]
    for i in pre_li:
        vec[i]=[]
        vec_woun[i]=[]
        vec_wosim[i]=[]

        vec_inclu[i] = []
        vec_woun_inclu[i] = []
        vec_wosim_inclu[i] = []



    print("time:", datetime.datetime.now())

    feature_mat, weight_state, unique_index_arr_id, stellar_bscov, stellar_btcov, rnntest_sc, nc_cov, nc_cam, rnntest_sc_cam, mymethod, trend_set, right, fault_types, deepgini, maxp = get_selection_information_new(
        file_path, model, lstm_classifier,
        dense_model, wrapper_path, w2v_path, time_steps)


    for pre in pre_li:
        select_num = int(total_num * 0.01 * pre)

        # selection
        mymethod_selected = total_selection(mymethod, total_num, select_num)
        mymethod_selected_wosim = total_selection_random_sim(mymethod, total_num, select_num)
        mymethod_selected_woun = total_selection_random_uncertainty(mymethod, total_num, select_num)


        vec_wosim_R, vec_wosim_P, _, _, _ = selection_evaluate(right, mymethod_selected_wosim)
        vec_woun_R, vec_woun_P, _, _, _ = selection_evaluate(right, mymethod_selected_woun)
        vec_R, vec_P, _, _, _ = selection_evaluate(right, mymethod_selected)

        vec_woun[pre].append(vec_woun_P)
        vec[pre].append(vec_P)
        vec_wosim[pre].append(vec_wosim_P)

        vec_woun_inclu[pre].append(vec_woun_R)
        vec_inclu[pre].append(vec_R)
        vec_wosim_inclu[pre].append(vec_wosim_R)

    result_dict = {'Vec_w10':vec[10],'Vec_w20':vec[20],
                   'Vec_un_w10': vec_woun[10], 'Vec_un_w20': vec_woun[20],
                   'Vec_sim_w10': vec_wosim[10], 'Vec_sim_w20': vec_wosim[20],
                   'Vec_inclu_w10': vec_inclu[10], 'Vec_inclu_w20': vec_inclu[20],
                   'Vec_un_inclu_w10': vec_woun_inclu[10], 'Vec_un_inclu_w20': vec_woun_inclu[20],
                   'Vec_sim_inclu_w10': vec_wosim_inclu[10], 'Vec_sim_inclu_w20': vec_wosim_inclu[20],
                   }

    print(result_dict)
    df = pd.DataFrame(result_dict)
    os.makedirs("./exp_results/ablation", exist_ok=True)
    df.to_csv("./exp_results/ablation/ablation_{}_{}.csv".format(args.dataset, args.model_type))

    print("Finished! The results are saved in: [./exp_results/ablation/ablation_{}_{}.csv]".format(args.dataset, args.model_type))
