import argparse
from statics import *
import numpy as np
import pandas as pd
import os
from selection_tools import get_selection_sensitive
import sys
import tensorflow as tf
import keras.backend.tensorflow_backend as K

# Specify that the first GPU is available, if there is no GPU, apply: "-1"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # Do not occupy all of the video memory, allocate on demand
sess = tf.compat.v1.Session(config=config)

K.set_session(sess)

# RQ: Sensitive Analize for tau
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

    else:
        print("The model and data set are incorrect.")
        sys.exit(1)

    mymethod, right, fault_types = get_selection_sensitive(file_path, model, lstm_classifier,dense_model, wrapper_path, w2v_path,time_steps)

    vec_0, vec_15, vec_30, vec_45, vec_60, vec_75, vec_90 = [], [], [], [], [], [], []
    vec_0_r, vec_15_r, vec_30_r, vec_45_r, vec_60_r, vec_75_r, vec_90_r = [], [], [], [], [], [], []
    f_0, f_15, f_30, f_45, f_60, f_75, f_90 = [], [], [], [], [], [], []

    uncertainty, diss_0, diss_15, diss_30, diss_45, diss_60, diss_75, diss_90 = mymethod[0], mymethod[1], mymethod[2], mymethod[3], mymethod[4], mymethod[5], mymethod[6], mymethod[7]

    # print("final:",mymethod)
    pre_li = [10, 20]
    for pre in pre_li:
        select_num = int(total_num * 0.01 * pre)

        mymethod_selected_0 = total_selection_sensitive([uncertainty, diss_0], total_num, select_num)
        mymethod_selected_15 = total_selection_sensitive([uncertainty, diss_15], total_num, select_num)
        mymethod_selected_30 = total_selection_sensitive([uncertainty, diss_30], total_num, select_num)
        mymethod_selected_45 = total_selection_sensitive([uncertainty, diss_45], total_num, select_num)
        mymethod_selected_60 = total_selection_sensitive([uncertainty, diss_60], total_num, select_num)
        mymethod_selected_75 = total_selection_sensitive([uncertainty, diss_75], total_num, select_num)
        mymethod_selected_90 = total_selection_sensitive([uncertainty, diss_90], total_num, select_num)


        Vec_0_R, Vec_0_P, _, _, _ = selection_evaluate(right, mymethod_selected_0)
        Vec_15_R, Vec_15_P, _, _, _ = selection_evaluate(right, mymethod_selected_15)
        Vec_30_R, Vec_30_P, _, _, _ = selection_evaluate(right, mymethod_selected_30)
        Vec_45_R, Vec_45_P, _, _, _ = selection_evaluate(right, mymethod_selected_45)
        Vec_60_R, Vec_60_P, _, _, _ = selection_evaluate(right, mymethod_selected_60)
        Vec_75_R, Vec_75_P, _, _, _ = selection_evaluate(right, mymethod_selected_75)
        Vec_90_R, Vec_90_P, _, _, _ = selection_evaluate(right, mymethod_selected_90)

        mymethod_selected_0 = total_selection([uncertainty, diss_0], total_num, select_num)
        mymethod_selected_15 = total_selection([uncertainty, diss_15], total_num, select_num)
        mymethod_selected_30 = total_selection([uncertainty, diss_30], total_num, select_num)
        mymethod_selected_45 = total_selection([uncertainty, diss_45], total_num, select_num)
        mymethod_selected_60 = total_selection([uncertainty, diss_60], total_num, select_num)
        mymethod_selected_75 = total_selection([uncertainty, diss_75], total_num, select_num)
        mymethod_selected_90 = total_selection([uncertainty, diss_90], total_num, select_num)

        Vec_0_fault = count_unique_elements(fault_types, mymethod_selected_0)
        Vec_15_fault = count_unique_elements(fault_types, mymethod_selected_15)
        Vec_30_fault = count_unique_elements(fault_types, mymethod_selected_30)
        Vec_45_fault = count_unique_elements(fault_types, mymethod_selected_45)
        Vec_60_fault = count_unique_elements(fault_types, mymethod_selected_60)
        Vec_75_fault = count_unique_elements(fault_types, mymethod_selected_75)
        Vec_90_fault = count_unique_elements(fault_types, mymethod_selected_90)

        f_0.append(Vec_0_fault)
        f_15.append(Vec_15_fault)
        f_30.append(Vec_30_fault)
        f_45.append(Vec_45_fault)
        f_60.append(Vec_60_fault)
        f_75.append(Vec_75_fault)
        f_90.append(Vec_90_fault)

        vec_0.append(Vec_0_P)
        vec_0_r.append(Vec_0_R)
        vec_15.append(Vec_15_P)
        vec_15_r.append(Vec_15_R)
        vec_30.append(Vec_30_P)
        vec_30_r.append(Vec_30_R)
        vec_45.append(Vec_45_P)
        vec_45_r.append(Vec_45_R)
        vec_60.append(Vec_60_P)
        vec_60_r.append(Vec_60_R)
        vec_75.append(Vec_75_P)
        vec_75_r.append(Vec_75_R)
        vec_90.append(Vec_90_P)
        vec_90_r.append(Vec_90_R)

    result_dict = {
        'vec_0': vec_0, 'vec_15': vec_15, 'vec_30': vec_30, 'vec_45': vec_45,
        'vec_60': vec_60, 'vec_75': vec_75, 'vec_90': vec_90,
        'vec_0_r': vec_0_r, 'vec_15_r': vec_15_r, 'vec_30_r': vec_30_r, 'vec_45_r': vec_45_r,
        'vec_60_r': vec_60_r, 'vec_75_r': vec_75_r, 'vec_90_r': vec_90_r,
        'f_0': f_0, 'f_15': f_15, 'f_30': f_30, 'f_45': f_45,
        'f_60': f_60, 'f_75': f_75, 'f_90': f_90,
    }


    print(result_dict)
    df = pd.DataFrame(result_dict)
    os.makedirs("./exp_results/sensitive", exist_ok=True)
    df.to_csv("./exp_results/sensitive/sensitive_{}_{}.csv".format(args.dataset, args.model_type))

    print("Finished! The results are saved in: [./exp_results/sensitive/sensitive_{}_{}.csv]".format(args.dataset, args.model_type))
