import os
import pickle
from deepstellar.Abstraction.StateAbstraction import StateAbstraction
from deepstellar.Abstraction.GraphWrapper import GraphWrapper
import argparse
import sys
sys.path.append("")


def get_abst_model(profile_save_path, abst_save_path, name_prefix, lstm_classifier, model):
    comp_num = 64  # 64
    k = 3
    m = 10
    bits = 8
    n_step = 0

    if not os.path.exists(profile_save_path):
        lstm_classifier.profile_train_data(model, profile_save_path)
        print("profiling done...")
    else:
        print("profiling is already done...")

    par_k = [m] * k
    stateAbst = StateAbstraction(profile_save_path, comp_num, bits, [m] * k, n_step)
    wrapper = GraphWrapper(stateAbst)
    wrapper.build_model()

    save_file = 'wrapper_%s_%s_%s.pkl' % (name_prefix, len(par_k), par_k[0])
    save_file = os.path.join(abst_save_path, save_file)
    os.makedirs(abst_save_path, exist_ok=True)
    with open(save_file, 'wb') as f:
        pickle.dump(wrapper, f)

    print('finish')


def mnist_lstm_abst():
    from RNNModels.mnist_demo.mnist_lstm import MnistLSTMClassifier
    dl_model = "./RNNModels/mnist_demo/models/mnist_lstm.h5"
    profile_save_path = "./RNNModels/mnist_demo/output/lstm/profile_save"
    abst_save_path = "./RNNModels/mnist_demo/output/lstm/abst_model"
    name_prefix = "lstm_mnist"
    lstm_classifier = MnistLSTMClassifier()
    model = lstm_classifier.load_hidden_state_model(dl_model)
    get_abst_model(profile_save_path, abst_save_path, name_prefix, lstm_classifier, model)


def mnist_blstm_abst():
    from RNNModels.mnist_demo.mnist_blstm import MnistBLSTMClassifier
    dl_model = "./RNNModels/mnist_demo/models/mnist_blstm.h5"
    profile_save_path = "./RNNModels/mnist_demo/output/blstm/profile_save"
    abst_save_path = "./RNNModels/mnist_demo/output/blstm/abst_model"
    name_prefix = "blstm_mnist"
    lstm_classifier = MnistBLSTMClassifier()
    model = lstm_classifier.load_hidden_state_model(dl_model)
    get_abst_model(profile_save_path, abst_save_path, name_prefix, lstm_classifier, model)


def snips_blstm_abst():
    from RNNModels.snips_demo.snips_blstm import SnipsBLSTMClassifier
    dl_model = "./RNNModels/snips_demo/models/snips_blstm.h5"
    profile_save_path = "./RNNModels/snips_demo/output/blstm/profile_save"
    abst_save_path = "./RNNModels/snips_demo/output/blstm/abst_model"
    name_prefix = "blstm_snips"
    lstm_classifier = SnipsBLSTMClassifier()
    lstm_classifier.embedding_path = "./RNNModels/snips_demo/save/embedding_matrix.npy"
    lstm_classifier.data_path = "./RNNModels/snips_demo/save/standard_data.npz"
    model = lstm_classifier.load_hidden_state_model(dl_model)
    get_abst_model(profile_save_path, abst_save_path, name_prefix, lstm_classifier, model)


def snips_gru_abst():
    from RNNModels.snips_demo.snips_gru import SnipsGRUClassifier
    dl_model = "./RNNModels/snips_demo/models/snips_gru.h5"
    profile_save_path = "./RNNModels/snips_demo/output/gru/profile_save"
    abst_save_path = "./RNNModels/snips_demo/output/gru/abst_model"
    name_prefix = "gru_snips"
    lstm_classifier = SnipsGRUClassifier()
    lstm_classifier.embedding_path = "./RNNModels/snips_demo/save/embedding_matrix.npy"
    lstm_classifier.data_path = "./RNNModels/snips_demo/save/standard_data.npz"
    model = lstm_classifier.load_hidden_state_model(dl_model)
    get_abst_model(profile_save_path, abst_save_path, name_prefix, lstm_classifier, model)


def fashion_lstm_abst():
    from RNNModels.fashion_demo.fashion_lstm import FashionLSTMClassifier
    dl_model = "./RNNModels/fashion_demo/models/fashion_lstm.h5"
    profile_save_path = "./RNNModels/fashion_demo/output/lstm/profile_save"
    abst_save_path = "./RNNModels/fashion_demo/output/lstm/abst_model"
    name_prefix = "lstm_fashion"
    lstm_classifier = FashionLSTMClassifier()
    model = lstm_classifier.load_hidden_state_model(dl_model)
    get_abst_model(profile_save_path, abst_save_path, name_prefix, lstm_classifier, model)


def fashion_gru_abst():
    from RNNModels.fashion_demo.fashion_gru import FashionGRUClassifier
    dl_model = "./RNNModels/fashion_demo/models/fashion_gru.h5"
    profile_save_path = "./RNNModels/fashion_demo/output/gru/profile_save"
    abst_save_path = "./RNNModels/fashion_demo/output/gru/abst_model"
    name_prefix = "gru_fashion"
    lstm_classifier = FashionGRUClassifier()
    model = lstm_classifier.load_hidden_state_model(dl_model)
    get_abst_model(profile_save_path, abst_save_path, name_prefix, lstm_classifier, model)


def agnews_lstm_abst():
    from RNNModels.agnews_demo.agnews_lstm import AGNewsLSTMClassifier
    dl_model = "./RNNModels/agnews_demo/models/agnews_lstm.h5"
    profile_save_path = "./RNNModels/agnews_demo/output/lstm/profile_save"
    abst_save_path = "./RNNModels/agnews_demo/output/lstm/abst_model"
    name_prefix = "lstm_agnews"
    lstm_classifier = AGNewsLSTMClassifier()
    lstm_classifier.embedding_path = "./RNNModels/agnews_demo/save/embedding_matrix.npy"
    lstm_classifier.data_path = "./RNNModels/agnews_demo/save/standard_data.npz"
    model = lstm_classifier.load_hidden_state_model(dl_model)
    get_abst_model(profile_save_path, abst_save_path, name_prefix, lstm_classifier, model)


def agnews_blstm_abst():
    from RNNModels.agnews_demo.agnews_blstm import AgnewsBLSTMClassifier
    dl_model = "./RNNModels/agnews_demo/models/agnews_blstm.h5"
    profile_save_path = "./RNNModels/agnews_demo/output/blstm/profile_save"
    abst_save_path = "./RNNModels/agnews_demo/output/blstm/abst_model"
    name_prefix = "blstm_agnews"
    lstm_classifier = AgnewsBLSTMClassifier()
    lstm_classifier.embedding_path = "./RNNModels/agnews_demo/save/embedding_matrix.npy"
    lstm_classifier.data_path = "./RNNModels/agnews_demo/save/standard_data.npz"
    model = lstm_classifier.load_hidden_state_model(dl_model)
    get_abst_model(profile_save_path, abst_save_path, name_prefix, lstm_classifier, model)


if __name__ == '__main__':
    parse = argparse.ArgumentParser(
        "Generate the abstract model for DeepStellar-cov.")
    parse.add_argument('-test_obj', required=True, choices=['mnist_lstm', 'mnist_blstm',
                                                            'snips_blstm', 'snips_gru',
                                                            'fashion_lstm', 'fashion_gru',
                                                            'agnews_lstm', 'agnews_blstm'])
    args = parse.parse_args()

    if args.test_obj == "mnist_lstm":
        mnist_lstm_abst()
    if args.test_obj == "mnist_blstm":
        mnist_blstm_abst()
    if args.test_obj == "snips_blstm":
        snips_blstm_abst()
    if args.test_obj == "snips_gru":
        snips_gru_abst()
    if args.test_obj == "fashion_lstm":
        fashion_lstm_abst()
    if args.test_obj == "fashion_gru":
        fashion_gru_abst()
    if args.test_obj == "agnews_lstm":
        agnews_lstm_abst()
    if args.test_obj == "agnews_blstm":
        agnews_blstm_abst()
