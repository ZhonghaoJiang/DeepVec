import pandas as pd
from gensim.models import Word2Vec
import re
from nltk.tokenize import word_tokenize
from keras.preprocessing.sequence import pad_sequences
import keras
import numpy as np
from statics import *
import sys
sys.path.append('..')

intent_dic = {"PlayMusic": 0, "AddToPlaylist": 1, "RateBook": 2, "SearchScreeningEvent": 3,
              "BookRestaurant": 4, "GetWeather": 5, "SearchCreativeWork": 6}


def process_snips_data(data_path, w2v_path):
    data = pd.read_csv(data_path)
    w2v_model = Word2Vec.load(w2v_path)
    sentences_ = list(data["text"])
    intent_ = list(data["intent"])
    intent = [intent_dic[i] for i in intent_]

    sentences = []
    for s in sentences_:
        clean = re.sub(r'[^ a-z A-Z 0-9]', " ", s)
        w = word_tokenize(clean)
        # stemming
        sentences.append([i.lower() for i in w])

    # 取得所有单词
    vocab_list = list(w2v_model.wv.vocab.keys())
    # 每个词语对应的索引
    word_index = {word: index for index, word in enumerate(vocab_list)}

    # 序列化
    def get_index(sentence):
        sequence = []
        for word in sentence:
            try:
                sequence.append(word_index[word])
            except KeyError:
                pass
        return sequence

    X_data = list(map(get_index, sentences))

    maxlen = 16  # 截长补短
    X_pad = pad_sequences(X_data, maxlen=maxlen)
    Y = keras.utils.to_categorical(intent, num_classes=7)
    return X_pad, Y


def process_agnews_data(data_path, w2v_path):
    data = pd.read_csv(data_path)
    w2v_model = Word2Vec.load(w2v_path)
    sentences_ = list(data["news"])
    intent_ = list(data["label"])
    intent = [i-1 for i in intent_]

    sentences = []
    for s in sentences_:
        clean = re.sub(r'[^ a-z A-Z 0-9]', " ", s)
        w = word_tokenize(clean)
        # stemming
        sentences.append([i.lower() for i in w])

    # 取得所有单词
    vocab_list = list(w2v_model.wv.vocab.keys())
    # 每个词语对应的索引
    word_index = {word: index for index, word in enumerate(vocab_list)}

    # 序列化
    def get_index(sentence):
        sequence = []
        for word in sentence:
            try:
                sequence.append(word_index[word])
            except KeyError:
                pass
        return sequence

    X_data = list(map(get_index, sentences))

    maxlen = 35  # 截长补短
    X_pad = pad_sequences(X_data, maxlen=maxlen)
    Y = keras.utils.to_categorical(intent, num_classes=4)
    return X_pad, Y


def get_selection_information(file_path, model, lstm_classifier, dense_model, wrapper_path, w2v_path, time_steps):
    if file_path.split(".")[-1] == "npz":
        with np.load(file_path, allow_pickle=True) as f:
            X, Y = f['X'], f['Y']
    if file_path.split(".")[-1] == "csv" and "snips" in file_path.split(".")[-2]:
        X, Y = process_snips_data(file_path, w2v_path)
    elif file_path.split(".")[-1] == "csv" and "agnews" in file_path.split(".")[-2]:
        X, Y = process_agnews_data(file_path, w2v_path)

    weight_state, stellar_bscov, stellar_btcov, rnntest_sc, rnntest_sc_cam, nc_cov, nc_cam, mymethod = [], [], [], [], [], [], [], []
    right, hscov_max_index, trend_set = [], [], []

    diss = []
    act_set = set()
    act_time_set = set()
    fault_types = []
    for idx, (x, y) in enumerate(zip(X, Y)):
        if file_path.split(".")[-1] == "npz" and "svhn" in file_path.split(".")[-2]:
            x_test = svhn_input_preprocess(np.array([x]))
            y_test = keras.utils.to_categorical(np.array([y]), num_classes=10)
        elif file_path.split(".")[-1] == "npz" and "svhn" not in file_path.split(".")[-2]:
            x_test = mnist_input_preprocess(np.array([x]))
            y_test = keras.utils.to_categorical(np.array([y]), num_classes=10)
        else:
            x_test = np.array([x])
            y_test = y

        classify_out_list, plus_sum, minus_sum = [], [], []
        lstm_out = model.predict(x_test)[1]

        # hs_cov
        hscov_max_index.append(np.argmax(lstm_out[0]))  # The index of the lstm_out matrix activated by this use case

        # nc_cov
        act = get_nc_activate(lstm_out)
        nc = len(act) / lstm_out[0].size if len(act) != 0 else 0
        nc_cov.append(nc)
        diff = act - act_set
        if not diff:  # True represents an empty set, i.e., there is no new act
            nc_cam.append(0)
        else:
            act_set = act_set.union(act)
            nc_cam.append(1)

        gini_seq = []
        vector_seq = []
        for i in range(time_steps):
            lstm_t = lstm_out[0][i]
            plus_sum.append(sum([lstm_ti for lstm_ti in lstm_t if lstm_ti > 0]))
            minus_sum.append(sum([lstm_ti for lstm_ti in lstm_t if lstm_ti < 0]))
            dense_array = lambda x: x[:, i, :]
            tmp = dense_model.predict(np.array(dense_array(lstm_out)))
            # print(tmp)
            vector_seq.append([item for sublist in tmp.tolist() for item in sublist])
            gini_seq.append(cal_distance(tmp))
            confident = np.max(tmp)
            # print(tmp)
            # exit(1)
            if confident >= 0.5 and i != (time_steps - 1):
                classify_out_list.append(np.argmax(tmp))
            # if i != (time_steps - 1):
            #     classify_out_list.append(np.argmax(tmp))
            elif i == (time_steps - 1):
                classify_out_list.append(np.argmax(tmp))


        mymethod.append(cal_weight_dis(np.array(gini_seq)))
        diss.append(deal_vecseq(vector_seq))




        trend_set.append(get_change_set(classify_out_list))
        weight_state.append(cacl_change_rate_with_weights(classify_out_list))

        # check the predict result is right or wrong
        check_predict_result(int(classify_out_list[-1]), int(np.argmax(y_test)), right)

        # Stellar Coverage
        BSCov, BTCov = get_stellar_cov(lstm_classifier, model, x, wrapper_path)
        stellar_bscov.append(BSCov)
        stellar_btcov.append(BTCov)

        # rnnTest Coverage
        SC, acted_time = get_testrnn_sc(plus_sum, minus_sum)
        rnntest_sc.append(SC)
        diff_acted_time = acted_time - act_time_set
        if not diff_acted_time:  # True represents an empty set, i.e., there is no new act
            rnntest_sc_cam.append(0)
        else:
            act_time_set = act_time_set.union(acted_time)
            rnntest_sc_cam.append(1)

        # Calculate fault types
        fault_type = (int(np.argmax(y_test)), int(classify_out_list[-1]))
        fault_types.append(fault_type)

        # print(classify_out_list)

    # hs_cov
    unique_index_arr, unique_index_arr_id = np.unique(hscov_max_index, return_index=True)

    final = [mymethod, diss]
    return weight_state, unique_index_arr_id, np.array(stellar_bscov), np.array(stellar_btcov), np.array(rnntest_sc), \
           np.array(nc_cov), np.array(nc_cam), np.array(rnntest_sc_cam), final, trend_set, right, fault_types

def calculate_fault_diversity(fault_types):
    unique_fault_types = set(fault_types)
    return len(unique_fault_types)

def get_selection_information_new(file_path, model, lstm_classifier, dense_model, wrapper_path, w2v_path, time_steps):
    if file_path.split(".")[-1] == "npz":
        with np.load(file_path, allow_pickle=True) as f:
            X, Y = f['X'], f['Y']
    if file_path.split(".")[-1] == "csv" and "snips" in file_path.split(".")[-2]:
        X, Y = process_snips_data(file_path, w2v_path)
    elif file_path.split(".")[-1] == "csv" and "agnews" in file_path.split(".")[-2]:
        X, Y = process_agnews_data(file_path, w2v_path)

    weight_state, stellar_bscov, stellar_btcov, rnntest_sc, rnntest_sc_cam, nc_cov, nc_cam, mymethod = [], [], [], [], [], [], [], []
    right, hscov_max_index, trend_set = [], [], []

    diss = []
    act_set = set()
    act_time_set = set()
    fault_types = []
    deepgini_set = []
    maxp_set=[]
    feature_mat=[]
    for idx, (x, y) in enumerate(zip(X, Y)):
        if file_path.split(".")[-1] == "npz" and "svhn" in file_path.split(".")[-2]:
            x_test = svhn_input_preprocess(np.array([x]))
            feature_mat.append(x_test)
            y_test = keras.utils.to_categorical(np.array([y]), num_classes=10)
        elif file_path.split(".")[-1] == "npz" and "svhn" not in file_path.split(".")[-2]:
            x_test = mnist_input_preprocess(np.array([x]))
            feature_mat.append(x_test)
            y_test = keras.utils.to_categorical(np.array([y]), num_classes=10)
        else:
            x_test = np.array([x])
            feature_mat.append(x_test)
            y_test = y

        classify_out_list, plus_sum, minus_sum = [], [], []
        lstm_out = model.predict(x_test)[1]

        # hs_cov
        hscov_max_index.append(np.argmax(lstm_out[0]))  # The index of the lstm_out matrix activated by this use case

        # nc_cov
        act = get_nc_activate(lstm_out)
        nc = len(act) / lstm_out[0].size if len(act) != 0 else 0
        nc_cov.append(nc)
        diff = act - act_set
        if not diff:  # True represents an empty set, i.e., there is no new act
            nc_cam.append(0)
        else:
            act_set = act_set.union(act)
            nc_cam.append(1)

        gini_seq = []
        vector_seq = []
        weighted_deepgini = 0
        weighted_maxp = 0
        weight_sum = 0

        for t in range(time_steps):
            lstm_t = lstm_out[0][t]
            plus_sum.append(sum([lstm_ti for lstm_ti in lstm_t if lstm_ti > 0]))
            minus_sum.append(sum([lstm_ti for lstm_ti in lstm_t if lstm_ti < 0]))
            dense_array = lambda x: x[:, t, :]
            tmp = dense_model.predict(np.array(dense_array(lstm_out)))
            vector_seq.append([item for sublist in tmp.tolist() for item in sublist])
            gini_seq.append(cal_distance(tmp))
            confident = np.max(tmp)
            # 计算权重
            weight = t ** 0.1
            weight_sum += weight
            deepgini = calculate_gini(tmp)
            maxp = calculate_maxp(tmp)
            weighted_deepgini += weight * deepgini
            weighted_maxp += weight * maxp
            if confident >= 0.5 and t != (time_steps - 1):
                classify_out_list.append(np.argmax(tmp))
            elif t == (time_steps - 1):
                classify_out_list.append(np.argmax(tmp))



        mymethod.append(cal_weight_dis(np.array(gini_seq)))
        diss.append(deal_vecseq(vector_seq))
        # 最终的deepgini和maxp值
        deepgini_set.append(weighted_deepgini / weight_sum)
        maxp_set.append(weighted_maxp / weight_sum)




        trend_set.append(get_change_set(classify_out_list))
        weight_state.append(cacl_change_rate_with_weights(classify_out_list))

        # check the predict result is right or wrong
        check_predict_result(int(classify_out_list[-1]), int(np.argmax(y_test)), right)

        # Stellar Coverage
        BSCov, BTCov = get_stellar_cov(lstm_classifier, model, x, wrapper_path)
        stellar_bscov.append(BSCov)
        stellar_btcov.append(BTCov)

        # rnnTest Coverage
        SC, acted_time = get_testrnn_sc(plus_sum, minus_sum)
        rnntest_sc.append(SC)
        diff_acted_time = acted_time - act_time_set
        if not diff_acted_time:  # True represents an empty set, i.e., there is no new act
            rnntest_sc_cam.append(0)
        else:
            act_time_set = act_time_set.union(acted_time)
            rnntest_sc_cam.append(1)

        # Calculate fault types
        fault_type = (int(np.argmax(y_test)), int(classify_out_list[-1]))
        fault_types.append(fault_type)

        # print(classify_out_list)

    # hs_cov
    unique_index_arr, unique_index_arr_id = np.unique(hscov_max_index, return_index=True)

    final = [mymethod, diss]
    return feature_mat, weight_state, unique_index_arr_id, np.array(stellar_bscov), np.array(stellar_btcov), np.array(rnntest_sc), \
           np.array(nc_cov), np.array(nc_cam), np.array(rnntest_sc_cam), final, trend_set, right, fault_types, deepgini_set, maxp_set

def deepvec_selection_information(file_path, model, lstm_classifier, dense_model, wrapper_path, w2v_path, time_steps):
    if file_path.split(".")[-1] == "npz":
        with np.load(file_path, allow_pickle=True) as f:
            X, Y = f['X'], f['Y']
    if file_path.split(".")[-1] == "csv" and "snips" in file_path.split(".")[-2]:
        X, Y = process_snips_data(file_path, w2v_path)
    elif file_path.split(".")[-1] == "csv" and "agnews" in file_path.split(".")[-2]:
        X, Y = process_agnews_data(file_path, w2v_path)

    weight_state, mymethod = [], []
    right, hscov_max_index, trend_set = [], [], []

    diss = []
    deepgini_set = []
    feature_mat=[]
    for idx, (x, y) in enumerate(zip(X, Y)):
        if file_path.split(".")[-1] == "npz" and "svhn" in file_path.split(".")[-2]:
            x_test = svhn_input_preprocess(np.array([x]))
            feature_mat.append(x_test)
            y_test = keras.utils.to_categorical(np.array([y]), num_classes=10)
        elif file_path.split(".")[-1] == "npz" and "svhn" not in file_path.split(".")[-2]:
            x_test = mnist_input_preprocess(np.array([x]))
            feature_mat.append(x_test)
            y_test = keras.utils.to_categorical(np.array([y]), num_classes=10)
        else:
            x_test = np.array([x])
            feature_mat.append(x_test)
            y_test = y

        classify_out_list, plus_sum, minus_sum = [], [], []
        lstm_out = model.predict(x_test)[1]

        gini_seq = []
        vector_seq = []
        weighted_deepgini = 0
        weighted_maxp = 0
        weight_sum = 0

        for t in range(time_steps):
            lstm_t = lstm_out[0][t]
            plus_sum.append(sum([lstm_ti for lstm_ti in lstm_t if lstm_ti > 0]))
            minus_sum.append(sum([lstm_ti for lstm_ti in lstm_t if lstm_ti < 0]))
            dense_array = lambda x: x[:, t, :]
            tmp = dense_model.predict(np.array(dense_array(lstm_out)))
            vector_seq.append([item for sublist in tmp.tolist() for item in sublist])
            gini_seq.append(cal_distance(tmp))
            confident = np.max(tmp)
            # 计算权重
            weight = t ** 0.1
            weight_sum += weight
            deepgini = calculate_gini(tmp)
            weighted_deepgini += weight * deepgini
            if confident >= 0.5 and t != (time_steps - 1):
                classify_out_list.append(np.argmax(tmp))
            elif t == (time_steps - 1):
                classify_out_list.append(np.argmax(tmp))



        mymethod.append(cal_weight_dis(np.array(gini_seq)))
        diss.append(deal_vecseq(vector_seq))
        # 最终的deepgini和maxp值
        deepgini_set.append(weighted_deepgini / weight_sum)


        trend_set.append(get_change_set(classify_out_list))
        weight_state.append(cacl_change_rate_with_weights(classify_out_list))

        # check the predict result is right or wrong
        check_predict_result(int(classify_out_list[-1]), int(np.argmax(y_test)), right)

    final = [mymethod, diss]
    return feature_mat, weight_state, final, trend_set, right, deepgini_set


def get_selected_data(file_path, selected_li, w2v_path):
    selected_id = np.where(selected_li == 1)[0]
    X_selected, Y_selected = [], []
    if file_path.split(".")[-1] == "npz":
        with np.load(file_path, allow_pickle=True) as f:
            X, Y = f['X'], f['Y']
        for idx in selected_id:
            X_selected.append(X[idx][0])
            Y_selected.append(Y[idx])

    elif file_path.split(".")[-1] == "csv" and "snips" in file_path.split(".")[-2]:
        X, Y = process_snips_data(file_path, w2v_path)
        for idx in selected_id:
            X_selected.append(np.array(X[idx]))
            Y_selected.append(Y[idx])

    elif file_path.split(".")[-1] == "csv" and "agnews" in file_path.split(".")[-2]:
        X, Y = process_agnews_data(file_path, w2v_path)
        for idx in selected_id:
            X_selected.append(np.array(X[idx]))
            Y_selected.append(Y[idx])

    X_selected_array = np.array(X_selected)
    Y_selected_array = np.array(Y_selected)
    return X_selected_array, Y_selected_array


def get_val_data(file_path, w2v_path):
    X_val, Y_val = [], []
    if file_path.split(".")[-1] == "npz":
        with np.load(file_path, allow_pickle=True) as f:
            X, Y = f['X'], f['Y']
        for x in X:
            X_val.append(x[0])
        return np.array(X_val), Y

    elif file_path.split(".")[-1] == "csv" and "snips" in file_path.split(".")[-2]:
        X, Y = process_snips_data(file_path, w2v_path)
        for x in X:
            X_val.append(x)
        return np.array(X_val), Y

    elif file_path.split(".")[-1] == "csv" and "agnews" in file_path.split(".")[-2]:
        X, Y = process_agnews_data(file_path, w2v_path)
        for x in X:
            X_val.append(x)
        return np.array(X_val), Y

def get_selection_sensitive(file_path, model, lstm_classifier, dense_model, wrapper_path, w2v_path, time_steps):
    if file_path.split(".")[-1] == "npz":
        with np.load(file_path, allow_pickle=True) as f:
            X, Y = f['X'], f['Y']
    if file_path.split(".")[-1] == "csv" and "snips" in file_path.split(".")[-2]:
        X, Y = process_snips_data(file_path, w2v_path)
    elif file_path.split(".")[-1] == "csv" and "agnews" in file_path.split(".")[-2]:
        X, Y = process_agnews_data(file_path, w2v_path)

    weight_state, stellar_bscov, stellar_btcov, rnntest_sc, rnntest_sc_cam, nc_cov, nc_cam, mymethod = [], [], [], [], [], [], [], []
    right, hscov_max_index, trend_set = [], [], []

    diss_0, diss_15, diss_30, diss_45, diss_60, diss_75, diss_90 = [], [], [], [], [], [], []
    fault_types=[]

    for idx, (x, y) in enumerate(zip(X, Y)):
        if file_path.split(".")[-1] == "npz" and "svhn" in file_path.split(".")[-2]:
            x_test = svhn_input_preprocess(np.array([x]))
            y_test = keras.utils.to_categorical(np.array([y]), num_classes=10)
        elif file_path.split(".")[-1] == "npz" and "svhn" not in file_path.split(".")[-2]:
            x_test = mnist_input_preprocess(np.array([x]))
            y_test = keras.utils.to_categorical(np.array([y]), num_classes=10)
        else:
            x_test = np.array([x])
            y_test = y

        classify_out_list, plus_sum, minus_sum = [], [], []
        lstm_out = model.predict(x_test)[1]

        gini_seq = []
        vector_seq = []
        for i in range(time_steps):
            lstm_t = lstm_out[0][i]
            plus_sum.append(sum([lstm_ti for lstm_ti in lstm_t if lstm_ti > 0]))
            minus_sum.append(sum([lstm_ti for lstm_ti in lstm_t if lstm_ti < 0]))
            dense_array = lambda x: x[:, i, :]
            tmp = dense_model.predict(np.array(dense_array(lstm_out)))
            vector_seq.append([item for sublist in tmp.tolist() for item in sublist])
            gini_seq.append(cal_distance(tmp))
            confident = np.max(tmp)
            if confident >= 0.5 and i != (time_steps - 1):
                classify_out_list.append(np.argmax(tmp))
            elif i == (time_steps - 1):
                classify_out_list.append(np.argmax(tmp))

        mymethod.append(cal_weight_dis(np.array(gini_seq)))
        diss_0.append(deal_vecseq_sensitive(vector_seq, 0))
        diss_15.append(deal_vecseq_sensitive(vector_seq, 15))
        diss_30.append(deal_vecseq_sensitive(vector_seq, 30))
        diss_45.append(deal_vecseq_sensitive(vector_seq, 45))
        diss_60.append(deal_vecseq_sensitive(vector_seq, 60))
        diss_75.append(deal_vecseq_sensitive(vector_seq, 75))
        diss_90.append(deal_vecseq_sensitive(vector_seq, 90))

        # check the predict result is right or wrong
        check_predict_result(int(classify_out_list[-1]), int(np.argmax(y_test)), right)

        # Calculate fault types
        fault_type = (int(np.argmax(y_test)), int(classify_out_list[-1]))
        fault_types.append(fault_type)


    final = [mymethod, diss_0, diss_15, diss_30, diss_45, diss_60, diss_75, diss_90]
    return final, right, fault_types


from ATS.ATS import ATS
def ats_selection_rank(file_path, w2v_path, model, dense_model):
    ats = ATS()
    if file_path.split(".")[-1] == "npz":
        with np.load(file_path, allow_pickle=True) as f:
            X, Y = f['X'], f['Y']
    if file_path.split(".")[-1] == "csv" and "snips" in file_path.split(".")[-2]:
        X, Y = process_snips_data(file_path, w2v_path)
        Y = np.argmax(Y, axis=1)
    elif file_path.split(".")[-1] == "csv" and "agnews" in file_path.split(".")[-2]:
        X, Y = process_agnews_data(file_path, w2v_path)
        Y = np.argmax(Y, axis=1)

    if file_path.split(".")[-1] == "npz" and "svhn" in file_path.split(".")[-2]:
        div_rank, _, _ = ats.get_priority_sequence(X, Y, 8, model,dense_model, th=0.001,dataset='svhn')
    elif file_path.split(".")[-1] == "npz" and "svhn" not in file_path.split(".")[-2]:
        div_rank, _, _ = ats.get_priority_sequence(X, Y, 8, model,dense_model, th=0.001,dataset='mnist')
    elif file_path.split(".")[-1] == "csv" and "snips" in file_path.split(".")[-2]:
        div_rank, _, _ = ats.get_priority_sequence(X, Y, 6, model,dense_model, th=0.001 ,dataset='snips')
    else:
        div_rank, _, _ = ats.get_priority_sequence(X, Y, 4, model,dense_model, th=0.001,dataset='agnews')
        #6644

    return div_rank

def ats_selection(rank ,length, selected_num):
    selected = np.zeros(length)
    selected[rank[:selected_num]] = 1
    return selected