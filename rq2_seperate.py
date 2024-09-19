import argparse
from statics import *
import numpy as np
import pandas as pd
import os
from selection_tools import get_selection_information, get_selection_information_new, ats_selection,ats_selection_rank
import sys
import tensorflow as tf
import keras.backend.tensorflow_backend as K

# Specify that the first GPU is available, if there is no GPU, apply: "-1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # Do not occupy all of the video memory, allocate on demand
sess = tf.compat.v1.Session(config=config)

K.set_session(sess)

# RQ2: The inclusiveness on {1%, 2%, ..., 40%} selected test set.
if __name__ == '__main__':
    parse = argparse.ArgumentParser("Calculate the inclusiveness for the selected dataset.")
    parse.add_argument('-dl_model', help='path of dl model', required=True)
    parse.add_argument('-model_type', required=True, choices=['lstm', 'blstm', 'gru'])
    parse.add_argument('-dataset', required=True, choices=['mnist', 'snips', 'fashion', 'agnews', 'svhn'])
    parse.add_argument('-mode', required=True, choices=['ori', 'aug'])
    args = parse.parse_args()

    if args.model_type == "lstm" and args.dataset == "mnist":
        time_steps = 28
        w2v_path = ""
        from RNNModels.mnist_demo.mnist_lstm import MnistLSTMClassifier

        lstm_classifier = MnistLSTMClassifier()
        model = lstm_classifier.load_hidden_state_model(args.dl_model)
        dense_classifier = MnistLSTMClassifier()
        dense_model = dense_classifier.reload_dense(args.dl_model)

        ori_path = "./gen_data/mnist_toselect/mnist_toselect_ori.npz"
        aug_path = "./gen_data/mnist_toselect/mnist_toselect_aug.npz"
        wrapper_path = "./RNNModels/mnist_demo/output/lstm/abst_model/wrapper_lstm_mnist_3_10.pkl"
        total_num = 3000

    elif args.model_type == "blstm" and args.dataset == "mnist":
        time_steps = 28
        w2v_path = ""
        from RNNModels.mnist_demo.mnist_blstm import MnistBLSTMClassifier

        lstm_classifier = MnistBLSTMClassifier()
        model = lstm_classifier.load_hidden_state_model(args.dl_model)
        dense_classifier = MnistBLSTMClassifier()
        dense_model = dense_classifier.reload_dense(args.dl_model)

        ori_path = "./gen_data/mnist_toselect/mnist_toselect_ori.npz"
        aug_path = "./gen_data/mnist_toselect/mnist_toselect_aug.npz"
        wrapper_path = "./RNNModels/mnist_demo/output/blstm/abst_model/wrapper_blstm_mnist_3_10.pkl"
        total_num = 3000

    elif args.model_type == "gru" and args.dataset == "mnist":
        time_steps = 28
        w2v_path = ""
        from RNNModels.mnist_demo.mnist_gru import MnistGRUClassifier

        lstm_classifier = MnistGRUClassifier()
        model = lstm_classifier.load_hidden_state_model(args.dl_model)
        dense_classifier = MnistGRUClassifier()
        dense_model = dense_classifier.reload_dense(args.dl_model)

        ori_path = "./gen_data/mnist_toselect/mnist_toselect_ori.npz"
        aug_path = "./gen_data/mnist_toselect/mnist_toselect_aug.npz"
        wrapper_path = "./RNNModels/mnist_demo/output/gru/abst_model/wrapper_gru_mnist_3_10.pkl"
        total_num = 3000

    elif args.model_type == "blstm" and args.dataset == "snips":
        time_steps = 16
        from RNNModels.snips_demo.snips_blstm import SnipsBLSTMClassifier

        lstm_classifier = SnipsBLSTMClassifier()
        lstm_classifier.data_path = "./RNNModels/snips_demo/save/standard_data.npz"
        lstm_classifier.embedding_path = "./RNNModels/snips_demo/save/embedding_matrix.npy"
        model = lstm_classifier.load_hidden_state_model(args.dl_model)
        dense_classifier = SnipsBLSTMClassifier()
        dense_model = dense_classifier.reload_dense(args.dl_model)

        ori_path = "./gen_data/snips_toselect/snips_toselect_ori.csv"
        aug_path = "./gen_data/snips_toselect/snips_toselect_aug.csv"
        wrapper_path = "./RNNModels/snips_demo/output/blstm/abst_model/wrapper_blstm_snips_3_10.pkl"
        w2v_path = "./RNNModels/snips_demo/save/w2v_model"
        total_num = 2498

    elif args.model_type == "gru" and args.dataset == "snips":
        time_steps = 16
        from RNNModels.snips_demo.snips_gru import SnipsGRUClassifier

        lstm_classifier = SnipsGRUClassifier()
        lstm_classifier.data_path = "./RNNModels/snips_demo/save/standard_data.npz"
        lstm_classifier.embedding_path = "./RNNModels/snips_demo/save/embedding_matrix.npy"
        model = lstm_classifier.load_hidden_state_model(args.dl_model)
        dense_classifier = SnipsGRUClassifier()
        dense_model = dense_classifier.reload_dense(args.dl_model)

        ori_path = "./gen_data/snips_toselect/snips_toselect_ori.csv"
        aug_path = "./gen_data/snips_toselect/snips_toselect_aug.csv"
        wrapper_path = "./RNNModels/snips_demo/output/gru/abst_model/wrapper_gru_snips_3_10.pkl"
        w2v_path = "./RNNModels/snips_demo/save/w2v_model"
        total_num = 2498

    elif args.model_type == "lstm" and args.dataset == "snips":
        time_steps = 16
        from RNNModels.snips_demo.snips_lstm import SnipsLSTMClassifier

        lstm_classifier = SnipsLSTMClassifier()
        lstm_classifier.data_path = "./RNNModels/snips_demo/save/standard_data.npz"
        lstm_classifier.embedding_path = "./RNNModels/snips_demo/save/embedding_matrix.npy"
        model = lstm_classifier.load_hidden_state_model(args.dl_model)
        dense_classifier = SnipsLSTMClassifier()
        dense_model = dense_classifier.reload_dense(args.dl_model)

        ori_path = "./gen_data/snips_toselect/snips_toselect_ori.csv"
        aug_path = "./gen_data/snips_toselect/snips_toselect_aug.csv"
        wrapper_path = "./RNNModels/snips_demo/output/lstm/abst_model/wrapper_lstm_snips_3_10.pkl"
        w2v_path = "./RNNModels/snips_demo/save/w2v_model"
        total_num = 2498

    elif args.model_type == "lstm" and args.dataset == "fashion":
        time_steps = 28
        w2v_path = ""
        from RNNModels.fashion_demo.fashion_lstm import FashionLSTMClassifier

        lstm_classifier = FashionLSTMClassifier()
        model = lstm_classifier.load_hidden_state_model(args.dl_model)
        dense_classifier = FashionLSTMClassifier()
        dense_model = dense_classifier.reload_dense(args.dl_model)
        ori_path = "./gen_data/fashion_toselect/fashion_toselect_ori.npz"
        aug_path = "./gen_data/fashion_toselect/fashion_toselect_aug.npz"

        wrapper_path = "./RNNModels/fashion_demo/output/lstm/abst_model/wrapper_lstm_fashion_3_10.pkl"
        total_num = 3000

    elif args.model_type == "blstm" and args.dataset == "fashion":
        time_steps = 28
        w2v_path = ""
        from RNNModels.fashion_demo.fashion_blstm import FashionBLSTMClassifier

        lstm_classifier = FashionBLSTMClassifier()
        model = lstm_classifier.load_hidden_state_model(args.dl_model)
        dense_classifier = FashionBLSTMClassifier()
        dense_model = dense_classifier.reload_dense(args.dl_model)

        ori_path = "./gen_data/fashion_toselect/fashion_toselect_ori.npz"
        aug_path = "./gen_data/fashion_toselect/fashion_toselect_aug.npz"
        wrapper_path = "./RNNModels/fashion_demo/output/blstm/abst_model/wrapper_blstm_fashion_3_10.pkl"
        total_num = 3000

    elif args.model_type == "gru" and args.dataset == "fashion":
        time_steps = 28
        w2v_path = ""
        from RNNModels.fashion_demo.fashion_gru import FashionGRUClassifier

        lstm_classifier = FashionGRUClassifier()
        model = lstm_classifier.load_hidden_state_model(args.dl_model)
        dense_classifier = FashionGRUClassifier()
        dense_model = dense_classifier.reload_dense(args.dl_model)

        ori_path = "./gen_data/fashion_toselect/fashion_toselect_ori.npz"
        aug_path = "./gen_data/fashion_toselect/fashion_toselect_aug.npz"
        wrapper_path = "./RNNModels/fashion_demo/output/gru/abst_model/wrapper_gru_fashion_3_10.pkl"
        total_num = 3000

    elif args.model_type == "lstm" and args.dataset == "svhn":
        time_steps = 32
        w2v_path = ""
        from RNNModels.svhn_demo.svhn_lstm import SvhnLSTMClassifier

        lstm_classifier = SvhnLSTMClassifier()
        model = lstm_classifier.load_hidden_state_model(args.dl_model)
        dense_classifier = SvhnLSTMClassifier()
        dense_model = dense_classifier.reload_dense(args.dl_model)
        ori_path = "./gen_data/svhn_toselect/svhn_toselect_ori.npz"
        aug_path = "./gen_data/svhn_toselect/svhn_toselect_aug.npz"

        wrapper_path = "./RNNModels/svhn_demo/output/lstm/abst_model/wrapper_lstm_svhn_3_10.pkl"
        total_num = 3500

    elif args.model_type == "blstm" and args.dataset == "svhn":
        time_steps = 32
        w2v_path = ""
        from RNNModels.svhn_demo.svhn_blstm import SvhnBLSTMClassifier

        lstm_classifier = SvhnBLSTMClassifier()
        model = lstm_classifier.load_hidden_state_model(args.dl_model)
        dense_classifier = SvhnBLSTMClassifier()
        dense_model = dense_classifier.reload_dense(args.dl_model)

        ori_path = "./gen_data/svhn_toselect/svhn_toselect_ori.npz"
        aug_path = "./gen_data/svhn_toselect/svhn_toselect_aug.npz"
        wrapper_path = "./RNNModels/svhn_demo/output/blstm/abst_model/wrapper_blstm_svhn_3_10.pkl"
        total_num = 3500

    elif args.model_type == "gru" and args.dataset == "svhn":
        time_steps = 32
        w2v_path = ""
        from RNNModels.svhn_demo.svhn_gru import SvhnGRUClassifier

        lstm_classifier = SvhnGRUClassifier()
        model = lstm_classifier.load_hidden_state_model(args.dl_model)
        dense_classifier = SvhnGRUClassifier()
        dense_model = dense_classifier.reload_dense(args.dl_model)

        ori_path = "./gen_data/svhn_toselect/svhn_toselect_ori.npz"
        aug_path = "./gen_data/svhn_toselect/svhn_toselect_aug.npz"
        wrapper_path = "./RNNModels/svhn_demo/output/gru/abst_model/wrapper_gru_svhn_3_10.pkl"
        total_num = 3500


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

        ori_path = "./gen_data/agnews_toselect/agnews_toselect_ori.csv"
        aug_path = "./gen_data/agnews_toselect/agnews_toselect_aug.csv"
        wrapper_path = "./RNNModels/agnews_demo/output/lstm/abst_model/wrapper_lstm_agnews_3_10.pkl"
        total_num = 7600

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
        ori_path = "./gen_data/agnews_toselect/agnews_toselect_ori.csv"
        aug_path = "./gen_data/agnews_toselect/agnews_toselect_aug.csv"
        wrapper_path = "./RNNModels/agnews_demo/output/blstm/abst_model/wrapper_blstm_agnews_3_10.pkl"
        total_num = 7600

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
        ori_path = "./gen_data/agnews_toselect/agnews_toselect_ori.csv"
        aug_path = "./gen_data/agnews_toselect/agnews_toselect_aug.csv"
        wrapper_path = "./RNNModels/agnews_demo/output/gru/abst_model/wrapper_gru_agnews_3_10.pkl"
        total_num = 7600

    else:
        print("The model and data set are incorrect.")
        sys.exit(1)

    if args.mode =='ori':
        feature_mat, weight_state, unique_index_arr_id, stellar_bscov, stellar_btcov, rnntest_sc, nc_cov, nc_cam, rnntest_sc_cam, mymethod, trend_set, right, fault_types, deepgini, maxp = get_selection_information_new(
            ori_path, model, lstm_classifier,
            dense_model, wrapper_path, w2v_path, time_steps)
        rank = ats_selection_rank(ori_path, w2v_path, model, dense_model)
    else:
        feature_mat, weight_state, unique_index_arr_id, stellar_bscov, stellar_btcov, rnntest_sc, nc_cov, nc_cam, rnntest_sc_cam, mymethod, trend_set, right, fault_types, deepgini, maxp = get_selection_information_new(
            aug_path, model, lstm_classifier,
            dense_model, wrapper_path, w2v_path, time_steps)
        rank = ats_selection_rank(aug_path, w2v_path, model, dense_model)
    state_w_in, ran_in, RNNTestcov_in, Stellarbscov_in, Stellarbtcov_in, sc_ctm_in, sc_cam_in, \
        nc_ctm_in, nc_cam_in, my_in, deepgini_in, kmedoids_in, mmdcritic_in, maxp_in = [], [], [], [], [], [], [], [], [], [], [], [], [],[]
    ats_in = []
    # print("final:",mymethod)

    pre_li=[20,40]
    for pre in pre_li:
        select_num = int(total_num * 0.01 * pre)

        # selection
        state_w_selected = selection(weight_state, trend_set, select_num)
        # random_selected = ran_selection(total_num, select_num)
        # cov_selected = cam_selection(unique_index_arr_id, total_num, select_num)
        # bscov_selected = ctm_selection(np.array(stellar_bscov), total_num, select_num)
        # btcov_selected = ctm_selection(np.array(stellar_btcov), total_num, select_num)
        # sc_ctm_selected = ctm_selection(np.array(rnntest_sc), total_num, select_num)
        # sc_cam_selected = nc_cam_selection(np.array(rnntest_sc_cam), total_num, select_num)
        # nc_ctm_selected = ctm_selection(np.array(nc_cov), total_num, select_num)
        # nc_cam_selected = nc_cam_selection(np.array(nc_cam), total_num, select_num)
        mymethod_selected = total_selection(mymethod, total_num, select_num)

        deepgini_selected = gini_selection(np.array(deepgini), total_num, select_num)
        # kmedoids_selected = kmedoids_selection(np.array(feature_mat), total_num, select_num)
        # mmdcritic_selected = mmdcritic_selection(np.array(feature_mat), total_num, select_num)
        # maxp_selected = maxP_selection(np.array(maxp), total_num, select_num)
        ats_selected = ats_selection(rank, total_num, select_num)

        state_w_R, state_w_P, _, _, _ = selection_evaluate(right, state_w_selected)
        # random_R, random_P, _, _, _ = selection_evaluate(right, random_selected)
        # cov_R, cov_P, _, _, _ = selection_evaluate(right, cov_selected)
        # bscov_R, bscov_P, _, _, _ = selection_evaluate(right, bscov_selected)
        # btcov_R, btcov_P, _, _, _ = selection_evaluate(right, btcov_selected)
        # sc_ctm_R, sc_ctm_P, _, _, _ = selection_evaluate(right, sc_ctm_selected)
        # sc_cam_R, sc_cam_P, _, _, _ = selection_evaluate(right, sc_cam_selected)
        # nc_ctm_R, nc_ctm_P, _, _, _ = selection_evaluate(right, nc_ctm_selected)
        # nc_cam_R, nc_cam_P, _, _, _ = selection_evaluate(right, nc_cam_selected)
        my_cam_R, my_cam_P, _, _, _ = selection_evaluate(right, mymethod_selected)

        gini_R, gini_P, _, _, _ = selection_evaluate(right, deepgini_selected)
        # kmedoids_R, kmedoids_P, _, _, _ = selection_evaluate(right, kmedoids_selected)
        # mmdcritic_R, mmdcritic_P, _, _, _ = selection_evaluate(right, mmdcritic_selected)
        # maxp_R, maxp_P, _, _, _ = selection_evaluate(right, deepgini_selected)
        ats_R, ats_P, _, _, _ = selection_evaluate(right, ats_selected)


        state_w_in.append(state_w_R)
        # ran_in.append(random_R)
        # RNNTestcov_in.append(cov_R)
        # Stellarbscov_in.append(bscov_R)
        # Stellarbtcov_in.append(btcov_R)
        # sc_ctm_in.append(sc_ctm_R)
        # sc_cam_in.append(sc_cam_R)
        # nc_ctm_in.append(nc_ctm_R)
        # nc_cam_in.append(nc_cam_R)
        my_in.append(my_cam_R)
        deepgini_in.append(gini_R)
        # kmedoids_in.append(kmedoids_R)
        # mmdcritic_in.append(mmdcritic_R)
        # maxp_in.append(maxp_R)
        ats_in.append(ats_R)

    # result_dict = {
    #     'state': state_w_in, 'random': ran_in, 'RNNTestcov': RNNTestcov_in, 'Stellarbscov': Stellarbscov_in,
    #     'Stellarbtcov': Stellarbtcov_in, 'testRNNsc': sc_ctm_in, 'testRNNsc_cam': sc_cam_in,
    #     'nc_ctm': nc_ctm_in, 'nc_cam': nc_cam_in, 'deepgini': deepgini_in,
    #     'kmedoids': kmedoids_in, 'mmdcritic': mmdcritic_in, 'maxp': maxp_in, 'ats':ats_in, 'Vec': my_in,
    # }

    result_dict = {
        'state': state_w_in, 'deepgini': deepgini_in,
        'ats': ats_in, 'Vec': my_in,
    }
    print(result_dict)
    df = pd.DataFrame(result_dict)
    os.makedirs("./exp_results/rq2_full", exist_ok=True)
    df.to_csv("./exp_results/rq2_full/rq2_{}_{}_{}.csv".format(args.dataset, args.model_type,args.mode))

    print("Finished! The results are saved in: [./exp_results/rq2_full/rq2_{}_{}_{}.csv]".format(args.dataset, args.model_type,args.mode))
