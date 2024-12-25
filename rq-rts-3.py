import argparse
import numpy as np
import pandas as pd
import os
from selection_tools import rts_selection_rank, rts_selection, get_val_data, get_selected_data
import sys
import tensorflow as tf
import keras.backend.tensorflow_backend as K

# Specify that the first GPU is available, if there is no GPU, apply: "-1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True   # Do not occupy all of the video memory, allocate on demand
sess = tf.compat.v1.Session(config=config)

K.set_session(sess)


# RQ3: Retrain the RNNs with selected data
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

        to_select_path = "./gen_data/mnist_retrain/mnist_toselect.npz"
        ori_val_path = "./gen_data/mnist_retrain/mnist_ori_test.npz"
        aug_val_path = "./gen_data/mnist_retrain/mnist_aug_test.npz"
        mix_val_path = "./gen_data/mnist_retrain/mnist_mix_test.npz"
        retrain_save_path = "./RNNModels/mnist_demo/models/lstm_selected_"
        wrapper_path = "./RNNModels/mnist_demo/output/lstm/abst_model/wrapper_lstm_mnist_3_10.pkl"
        total_num = 16000

    elif args.model_type == "blstm" and args.dataset == "mnist":
        time_steps = 28
        w2v_path = ""
        from RNNModels.mnist_demo.mnist_blstm import MnistBLSTMClassifier

        lstm_classifier = MnistBLSTMClassifier()
        model = lstm_classifier.load_hidden_state_model(args.dl_model)
        dense_classifier = MnistBLSTMClassifier()
        dense_model = dense_classifier.reload_dense(args.dl_model)

        to_select_path = "./gen_data/mnist_retrain/mnist_toselect.npz"
        ori_val_path = "./gen_data/mnist_retrain/mnist_ori_test.npz"
        aug_val_path = "./gen_data/mnist_retrain/mnist_aug_test.npz"
        mix_val_path = "./gen_data/mnist_retrain/mnist_mix_test.npz"
        retrain_save_path = "./RNNModels/mnist_demo/models/blstm_selected_"
        wrapper_path = "./RNNModels/mnist_demo/output/blstm/abst_model/wrapper_blstm_mnist_3_10.pkl"
        total_num = 16000

    elif args.model_type == "gru" and args.dataset == "mnist":
        time_steps = 28
        w2v_path = ""
        from RNNModels.mnist_demo.mnist_gru import MnistGRUClassifier

        lstm_classifier = MnistGRUClassifier()
        model = lstm_classifier.load_hidden_state_model(args.dl_model)
        dense_classifier = MnistGRUClassifier()
        dense_model = dense_classifier.reload_dense(args.dl_model)

        to_select_path = "./gen_data/mnist_retrain/mnist_toselect.npz"
        ori_val_path = "./gen_data/mnist_retrain/mnist_ori_test.npz"
        aug_val_path = "./gen_data/mnist_retrain/mnist_aug_test.npz"
        mix_val_path = "./gen_data/mnist_retrain/mnist_mix_test.npz"
        retrain_save_path = "./RNNModels/mnist_demo/models/gru_selected_"
        wrapper_path = "./RNNModels/mnist_demo/output/gru/abst_model/wrapper_gru_mnist_3_10.pkl"
        total_num = 16000



    elif args.model_type == "lstm" and args.dataset == "fashion":
        time_steps = 28
        w2v_path = ""
        from RNNModels.fashion_demo.fashion_lstm import FashionLSTMClassifier

        lstm_classifier = FashionLSTMClassifier()
        model = lstm_classifier.load_hidden_state_model(args.dl_model)
        dense_classifier = FashionLSTMClassifier()
        dense_model = dense_classifier.reload_dense(args.dl_model)

        to_select_path = "./gen_data/fashion_retrain/fashion_toselect.npz"
        ori_val_path = "./gen_data/fashion_retrain/fashion_ori_test.npz"
        aug_val_path = "./gen_data/fashion_retrain/fashion_aug_test.npz"
        mix_val_path = "./gen_data/fashion_retrain/fashion_mix_test.npz"
        retrain_save_path = "./RNNModels/fashion_demo/models/lstm_selected_"
        wrapper_path = "./RNNModels/fashion_demo/output/lstm/abst_model/wrapper_lstm_fashion_3_10.pkl"
        total_num = 16000

    elif args.model_type == "gru" and args.dataset == "fashion":
        time_steps = 28
        w2v_path = ""
        from RNNModels.fashion_demo.fashion_gru import FashionGRUClassifier

        lstm_classifier = FashionGRUClassifier()
        model = lstm_classifier.load_hidden_state_model(args.dl_model)
        dense_classifier = FashionGRUClassifier()
        dense_model = dense_classifier.reload_dense(args.dl_model)

        to_select_path = "./gen_data/fashion_retrain/fashion_toselect.npz"
        ori_val_path = "./gen_data/fashion_retrain/fashion_ori_test.npz"
        aug_val_path = "./gen_data/fashion_retrain/fashion_aug_test.npz"
        mix_val_path = "./gen_data/fashion_retrain/fashion_mix_test.npz"
        retrain_save_path = "./RNNModels/fashion_demo/models/gru_selected_"
        wrapper_path = "./RNNModels/fashion_demo/output/gru/abst_model/wrapper_gru_fashion_3_10.pkl"
        total_num = 16000

    elif args.model_type == "blstm" and args.dataset == "fashion":
        time_steps = 28
        w2v_path = ""
        from RNNModels.fashion_demo.fashion_blstm import FashionBLSTMClassifier

        lstm_classifier = FashionBLSTMClassifier()
        model = lstm_classifier.load_hidden_state_model(args.dl_model)
        dense_classifier = FashionBLSTMClassifier()
        dense_model = dense_classifier.reload_dense(args.dl_model)

        to_select_path = "./gen_data/fashion_retrain/fashion_toselect.npz"
        ori_val_path = "./gen_data/fashion_retrain/fashion_ori_test.npz"
        aug_val_path = "./gen_data/fashion_retrain/fashion_aug_test.npz"
        mix_val_path = "./gen_data/fashion_retrain/fashion_mix_test.npz"
        retrain_save_path = "./RNNModels/fashion_demo/models/blstm_selected_"
        wrapper_path = "./RNNModels/fashion_demo/output/blstm/abst_model/wrapper_blstm_fashion_3_10.pkl"
        total_num = 16000


    elif args.model_type == "lstm" and args.dataset == "svhn":
        time_steps = 32
        w2v_path = ""
        from RNNModels.svhn_demo.svhn_lstm import SvhnLSTMClassifier

        lstm_classifier = SvhnLSTMClassifier()
        model = lstm_classifier.load_hidden_state_model(args.dl_model)
        dense_classifier = SvhnLSTMClassifier()
        dense_model = dense_classifier.reload_dense(args.dl_model)

        to_select_path = "./gen_data/svhn_retrain/svhn_toselect.npz"
        ori_val_path = "./gen_data/svhn_retrain/svhn_ori_test.npz"
        aug_val_path = "./gen_data/svhn_retrain/svhn_aug_test.npz"
        mix_val_path = "./gen_data/svhn_retrain/svhn_mix_test.npz"
        retrain_save_path = "./RNNModels/svhn_demo/models/lstm_selected_"
        wrapper_path = "./RNNModels/svhn_demo/output/lstm/abst_model/wrapper_lstm_svhn_3_10.pkl"
        total_num = 21000

    elif args.model_type == "blstm" and args.dataset == "svhn":
        time_steps = 32
        w2v_path = ""
        from RNNModels.svhn_demo.svhn_blstm import SvhnBLSTMClassifier

        lstm_classifier = SvhnBLSTMClassifier()
        model = lstm_classifier.load_hidden_state_model(args.dl_model)
        dense_classifier = SvhnBLSTMClassifier()
        dense_model = dense_classifier.reload_dense(args.dl_model)

        to_select_path = "./gen_data/svhn_retrain/svhn_toselect.npz"
        ori_val_path = "./gen_data/svhn_retrain/svhn_ori_test.npz"
        aug_val_path = "./gen_data/svhn_retrain/svhn_aug_test.npz"
        mix_val_path = "./gen_data/svhn_retrain/svhn_mix_test.npz"
        retrain_save_path = "./RNNModels/svhn_demo/models/blstm_selected_"
        wrapper_path = "./RNNModels/svhn_demo/output/blstm/abst_model/wrapper_blstm_svhn_3_10.pkl"
        total_num = 21000

    elif args.model_type == "gru" and args.dataset == "svhn":
        time_steps = 32
        w2v_path = ""
        from RNNModels.svhn_demo.svhn_gru import SvhnGRUClassifier

        lstm_classifier = SvhnGRUClassifier()
        model = lstm_classifier.load_hidden_state_model(args.dl_model)
        dense_classifier = SvhnGRUClassifier()
        dense_model = dense_classifier.reload_dense(args.dl_model)

        to_select_path = "./gen_data/svhn_retrain/svhn_toselect.npz"
        ori_val_path = "./gen_data/svhn_retrain/svhn_ori_test.npz"
        aug_val_path = "./gen_data/svhn_retrain/svhn_aug_test.npz"
        mix_val_path = "./gen_data/svhn_retrain/svhn_mix_test.npz"
        retrain_save_path = "./RNNModels/svhn_demo/models/gru_selected_"
        wrapper_path = "./RNNModels/svhn_demo/output/gru/abst_model/wrapper_gru_svhn_3_10.pkl"
        total_num = 21000


    else:
        print("The model and data set are incorrect.")
        sys.exit(1)

    ori_acc_save, aug_acc_save, mix_acc_save = {}, {}, {}

    pre_li = [10, 15, 20, 25]

    rts_rank = rts_selection_rank(to_select_path, model)

    select_method = ['rts_selected']
    for item in select_method:
        ori_acc_save[item] = []
        aug_acc_save[item] = []
        mix_acc_save[item] = []

    for pre in pre_li:
        select_num = int(total_num * 0.01 * pre)

        # selection
        rts_selected = rts_selection(rts_rank, total_num, select_num)

        x_ori_val, y_ori_val = get_val_data(ori_val_path, w2v_path)
        x_aug_val, y_aug_val = get_val_data(aug_val_path, w2v_path)
        x_mix_val, y_mix_val = get_val_data(mix_val_path, w2v_path)

        for method_item in select_method:
            X_selected_array, Y_selected_array = get_selected_data(to_select_path, np.array(eval(method_item)), w2v_path)
            print("len(X_selected_array):", len(X_selected_array))
            retrained_model_path = retrain_save_path + str(pre) + "/" + str(method_item) + "_" + \
                                   str(args.dataset) + "_" + str(args.model_type) + ".h5"
            if not os.path.isfile(retrained_model_path):  # Has not been saved, needs to be trained
                os.makedirs(retrain_save_path + str(pre), exist_ok=True)
                lstm_classifier.retrain(X_selected_array, Y_selected_array, x_aug_val, y_aug_val, retrained_model_path)

            K.clear_session()

            ori_acc_tmp, ori_imp_tmp = lstm_classifier.evaluate_retrain(retrained_model_path, args.dl_model, x_ori_val, y_ori_val)
            aug_acc_tmp, aug_imp_tmp = lstm_classifier.evaluate_retrain(retrained_model_path, args.dl_model, x_aug_val, y_aug_val)
            mix_acc_tmp, mix_imp_tmp = lstm_classifier.evaluate_retrain(retrained_model_path, args.dl_model, x_mix_val, y_mix_val)

            ori_acc_save[method_item].append(
                str(round(ori_acc_tmp * 100, 2)) + " (" + str(round(ori_imp_tmp * 100, 2)) + "%)")
            aug_acc_save[method_item].append(
                str(round(aug_acc_tmp * 100, 2)) + " (" + str(round(aug_imp_tmp * 100, 2)) + "%)")
            mix_acc_save[method_item].append(
                str(round(mix_acc_tmp * 100, 2)) + " (" + str(round(mix_imp_tmp * 100, 2)) + "%)")

            print("{}_{}: ".format("ori_acc_imp", method_item), round(ori_imp_tmp * 100, 2))
            print("{}_{}: ".format("aug_acc_imp", method_item), round(aug_imp_tmp * 100, 2))
            print("{}_{}: ".format("mix_acc_imp", method_item), round(mix_imp_tmp * 100, 2))


    # ======== final result ========
    result_dict = {}
    result_dict['select rate'] = pre_li
    for method_item in select_method:
        result_dict[str(method_item) + str("_ori")] = ori_acc_save[method_item]
        result_dict[str(method_item) + str("_aug")] = aug_acc_save[method_item]
        result_dict[str(method_item) + str("_mix")] = mix_acc_save[method_item]

    print(result_dict)
    df = pd.DataFrame(result_dict)
    os.makedirs("./exp_results/rq-rts-3", exist_ok=True)
    df.to_csv("./exp_results/rq-rts-3/rq-rts-3_{}_{}.csv".format(args.dataset, args.model_type))

    print("Finished! The results are saved in: [./exp_results/rq-rts-3/rq-rts-3_{}_{}.csv]".format(args.dataset, args.model_type))
