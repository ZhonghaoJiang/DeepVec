import argparse
import os
import numpy as np
import pandas as pd


def gen_fashion():
    train_x_aug_path = "./dau/fashion_harder/x_train_aug.npy"
    train_y_aug_path = "./dau/fashion_harder/y_train_aug.npy"
    train_x_ori_path = "./dau/fashion_harder/x_train_ori.npy"
    train_y_ori_path = "./dau/fashion_harder/y_train_ori.npy"
    test_x_aug_path = "./dau/fashion_harder/x_test_aug.npy"
    test_y_aug_path = "./dau/fashion_harder/y_test_aug.npy"
    test_x_ori_path = "./dau/fashion_harder/x_test_ori.npy"
    test_y_ori_path = "./dau/fashion_harder/y_test_ori.npy"

    train_x_aug_data = np.load(train_x_aug_path)
    train_y_aug_data = np.load(train_y_aug_path)
    train_x_ori_data = np.load(train_x_ori_path)
    train_y_ori_data = np.load(train_y_ori_path)
    test_x_aug_data = np.load(test_x_aug_path)
    test_y_aug_data = np.load(test_y_aug_path)
    test_x_ori_data = np.load(test_x_ori_path)
    test_y_ori_data = np.load(test_y_ori_path)

    # to select data set
    x_to_select_mnist, y_to_select_mnist = [], []
    x_retrain_ori_test, y_retrain_ori_test = [], []
    x_retrain_aug_test, y_retrain_aug_test = [], []
    x_retrain_mix_test, y_retrain_mix_test = [], []
    aug_or_ori = []  # aug: 1, ori: 0

    # The total of train is 6k, plus 6k amplified samples to form to_select_dataset
    for i in range(len(train_x_ori_data)):  # 12000
        d1 = train_x_ori_data[i].reshape(1, 28, 28)
        x_to_select_mnist.append(d1)
        y_to_select_mnist.append(train_y_ori_data[i])
        aug_or_ori.append(0)

    for j in range(4000):  # 12000, select 6000
        d2 = train_x_aug_data[6000+j].reshape(1, 28, 28)
        x_to_select_mnist.append(d2)
        y_to_select_mnist.append(train_y_aug_data[6000+j])
        aug_or_ori.append(1)

    x_save_data = np.array(x_to_select_mnist)
    y_save_data = np.array(y_to_select_mnist)
    aug_or_ori = np.array(aug_or_ori)
    state = np.random.get_state()
    np.random.shuffle(x_save_data)
    np.random.set_state(state)
    np.random.shuffle(y_save_data)
    np.random.set_state(state)
    np.random.shuffle(aug_or_ori)

    for k in range(len(test_x_ori_data)):
        dd1 = test_x_ori_data[k].reshape(1, 28, 28)
        x_retrain_mix_test.append(dd1)
        y_retrain_mix_test.append(test_y_ori_data[k])
        x_retrain_ori_test.append(dd1)
        y_retrain_ori_test.append(test_y_ori_data[k])

    for p in range(len(test_x_aug_data)):
        dd2 = test_x_aug_data[p].reshape(1, 28, 28)
        x_retrain_mix_test.append(dd2)
        y_retrain_mix_test.append(test_y_aug_data[p])
        x_retrain_aug_test.append(dd2)
        y_retrain_aug_test.append(test_y_aug_data[p])

    os.makedirs("../../gen_data/fashion_retrain", exist_ok=True)
    np.savez("../../gen_data/fashion_retrain/fashion_toselect", X=x_save_data, Y=y_save_data)
    np.save("../../gen_data/fashion_retrain/data_type", aug_or_ori)
    np.savez("../../gen_data/fashion_retrain/fashion_ori_test", X=x_retrain_ori_test, Y=y_retrain_ori_test)
    np.savez("../../gen_data/fashion_retrain/fashion_aug_test", X=x_retrain_aug_test, Y=y_retrain_aug_test)
    np.savez("../../gen_data/fashion_retrain/fashion_mix_test", X=x_retrain_mix_test, Y=y_retrain_mix_test)


def gen_mnist():
    train_x_aug_path = "./dau/mnist_harder/x_train_aug.npy"
    train_y_aug_path = "./dau/mnist_harder/y_train_aug.npy"
    train_x_ori_path = "./dau/mnist_harder/x_train_ori.npy"
    train_y_ori_path = "./dau/mnist_harder/y_train_ori.npy"
    test_x_aug_path = "./dau/mnist_harder/x_test_aug.npy"
    test_y_aug_path = "./dau/mnist_harder/y_test_aug.npy"
    test_x_ori_path = "./dau/mnist_harder/x_test_ori.npy"
    test_y_ori_path = "./dau/mnist_harder/y_test_ori.npy"

    train_x_aug_data = np.load(train_x_aug_path)
    train_y_aug_data = np.load(train_y_aug_path)
    train_x_ori_data = np.load(train_x_ori_path)
    train_y_ori_data = np.load(train_y_ori_path)
    test_x_aug_data = np.load(test_x_aug_path)
    test_y_aug_data = np.load(test_y_aug_path)
    test_x_ori_data = np.load(test_x_ori_path)
    test_y_ori_data = np.load(test_y_ori_path)

    # to select data set
    x_to_select_mnist, y_to_select_mnist = [], []
    x_retrain_ori_test, y_retrain_ori_test = [], []
    x_retrain_aug_test, y_retrain_aug_test = [], []
    x_retrain_mix_test, y_retrain_mix_test = [], []
    aug_or_ori = []  # aug: 1, ori: 0

    # The total of train is 6k, plus 6k amplified samples to form to_select_dataset
    for i in range(len(train_x_ori_data)):  # 12000
        d1 = train_x_ori_data[i].reshape(1, 28, 28)
        x_to_select_mnist.append(d1)
        y_to_select_mnist.append(train_y_ori_data[i])
        aug_or_ori.append(0)

    for j in range(4000):  # 12000, select 6000
        d2 = train_x_aug_data[6000 + j].reshape(1, 28, 28)
        x_to_select_mnist.append(d2)
        y_to_select_mnist.append(train_y_aug_data[6000 + j])
        aug_or_ori.append(1)

    x_save_data = np.array(x_to_select_mnist)
    y_save_data = np.array(y_to_select_mnist)
    aug_or_ori = np.array(aug_or_ori)
    state = np.random.get_state()
    np.random.shuffle(x_save_data)
    np.random.set_state(state)
    np.random.shuffle(y_save_data)
    np.random.set_state(state)
    np.random.shuffle(aug_or_ori)

    for k in range(len(test_x_ori_data)):
        dd1 = test_x_ori_data[k].reshape(1, 28, 28)
        x_retrain_mix_test.append(dd1)
        y_retrain_mix_test.append(test_y_ori_data[k])
        x_retrain_ori_test.append(dd1)
        y_retrain_ori_test.append(test_y_ori_data[k])

    for p in range(len(test_x_aug_data)):
        dd2 = test_x_aug_data[p].reshape(1, 28, 28)
        x_retrain_mix_test.append(dd2)
        y_retrain_mix_test.append(test_y_aug_data[p])
        x_retrain_aug_test.append(dd2)
        y_retrain_aug_test.append(test_y_aug_data[p])

    os.makedirs("../../gen_data/mnist_retrain", exist_ok=True)
    np.savez("../../gen_data/mnist_retrain/mnist_toselect", X=x_save_data, Y=y_save_data)
    np.save("../../gen_data/mnist_retrain/data_type", aug_or_ori)
    np.savez("../../gen_data/mnist_retrain/mnist_ori_test", X=x_retrain_ori_test, Y=y_retrain_ori_test)
    np.savez("../../gen_data/mnist_retrain/mnist_aug_test", X=x_retrain_aug_test, Y=y_retrain_aug_test)
    np.savez("../../gen_data/mnist_retrain/mnist_mix_test", X=x_retrain_mix_test, Y=y_retrain_mix_test)


def gen_snips():
    data = pd.read_csv("../gen_test_dataset/dau/snips_harder/snips_toselect.csv")
    length = int(len(data) / 2)
    print("ori/aug length:", length)
    ori = data[:int(len(data) / 2)]  # The front half is ori, the back half is aug

    text_ori, intent_ori = [], []
    text_aug, intent_aug = [], []
    text_mix, intent_mix = [], []
    ori_test = pd.DataFrame(columns=('text', 'intent'))
    aug_test = pd.DataFrame(columns=('text', 'intent'))
    mix_test = pd.DataFrame(columns=('text', 'intent'))



    for i in range(len(ori)):
        text_ori.append(data.text[i])
        text_aug.append(data.text[i + length])
        intent_ori.append(data.intent[i])
        intent_aug.append(data.intent[i + length])

        text_mix.append(data.text[i])
        text_mix.append(data.text[i + length])
        intent_mix.append(data.intent[i])
        intent_mix.append(data.intent[i + length])


    for idx, (text_i, intent_i) in enumerate(zip(text_ori, intent_ori)):
        tmp = {'text': text_i, 'intent': intent_i}
        ori_test.loc[idx] = tmp
    for idx, (text_i, intent_i) in enumerate(zip(text_aug, intent_aug)):
        tmp = {'text': text_i, 'intent': intent_i}
        aug_test.loc[idx] = tmp
    for idx, (text_i, intent_i) in enumerate(zip(text_mix, intent_mix)):
        tmp = {'text': text_i, 'intent': intent_i}
        mix_test.loc[idx] = tmp

    os.makedirs("../../gen_data/snips_retrain", exist_ok=True)
    data.to_csv("../../gen_data/snips_retrain/snips_toselect.csv", index=False)
    ori_test.to_csv("../../gen_data/snips_retrain/snips_ori_test.csv")
    aug_test.to_csv("../../gen_data/snips_retrain/snips_aug_test.csv")
    mix_test.to_csv("../../gen_data/snips_retrain/snips_mix_test.csv")


def gen_agnews():
    data = pd.read_csv("../gen_test_dataset/dau/agnews_harder/agnews_toselect.csv")  # test + test_aug, 15200
    length = int(len(data) / 2)
    print("ori/aug length:", length)
    ori = data[:int(len(data) / 2)]  # The front half is ori, the back half is aug

    text_ori, intent_ori = [], []
    text_aug, intent_aug = [], []
    text_mix, intent_mix = [], []
    ori_test = pd.DataFrame(columns=('news', 'label'))
    aug_test = pd.DataFrame(columns=('news', 'label'))
    mix_test = pd.DataFrame(columns=('news', 'label'))

    for i in range(len(ori)):
        text_ori.append(data.news[i])
        text_aug.append(data.news[i + length])
        intent_ori.append(data.label[i])
        intent_aug.append(data.label[i + length])

        text_mix.append(data.news[i])
        text_mix.append(data.news[i + length])
        intent_mix.append(data.label[i])
        intent_mix.append(data.label[i + length])

    for idx, (text_i, intent_i) in enumerate(zip(text_ori, intent_ori)):
        tmp = {'news': text_i, 'label': intent_i}
        ori_test.loc[idx] = tmp
    for idx, (text_i, intent_i) in enumerate(zip(text_aug, intent_aug)):
        tmp = {'news': text_i, 'label': intent_i}
        aug_test.loc[idx] = tmp
    for idx, (text_i, intent_i) in enumerate(zip(text_mix, intent_mix)):
        tmp = {'news': text_i, 'label': intent_i}
        mix_test.loc[idx] = tmp

    os.makedirs("../../gen_data/agnews_retrain", exist_ok=True)
    data.to_csv("../../gen_data/agnews_retrain/agnews_toselect.csv", index=False)
    ori_test.to_csv("../../gen_data/agnews_retrain/agnews_ori_test.csv")
    aug_test.to_csv("../../gen_data/agnews_retrain/agnews_aug_test.csv")
    mix_test.to_csv("../../gen_data/agnews_retrain/agnews_mix_test.csv")

def gen_svhn():
    train_x_aug_path = "./dau/svhn_harder/x_train_aug.npy"
    train_y_aug_path = "./dau/svhn_harder/y_train_aug.npy"
    train_x_ori_path = "./dau/svhn_harder/x_train_ori.npy"
    train_y_ori_path = "./dau/svhn_harder/y_train_ori.npy"
    test_x_aug_path = "./dau/svhn_harder/x_test_aug.npy"
    test_y_aug_path = "./dau/svhn_harder/y_test_aug.npy"
    test_x_ori_path = "./dau/svhn_harder/x_test_ori.npy"
    test_y_ori_path = "./dau/svhn_harder/y_test_ori.npy"

    train_x_aug_data = np.load(train_x_aug_path)
    train_y_aug_data = np.load(train_y_aug_path)
    train_x_ori_data = np.load(train_x_ori_path)
    train_y_ori_data = np.load(train_y_ori_path)
    test_x_aug_data = np.load(test_x_aug_path)
    test_y_aug_data = np.load(test_y_aug_path)
    test_x_ori_data = np.load(test_x_ori_path)
    test_y_ori_data = np.load(test_y_ori_path)

    # to select data set
    x_to_select_svhn, y_to_select_svhn = [], []
    x_retrain_ori_test, y_retrain_ori_test = [], []
    x_retrain_aug_test, y_retrain_aug_test = [], []
    x_retrain_mix_test, y_retrain_mix_test = [], []
    aug_or_ori = []  # aug: 1, ori: 0

    # The total of train is 6k, plus 6k amplified samples to form to_select_dataset
    for i in range(len(train_x_ori_data)):  # 14000
        d1 = train_x_ori_data[i].reshape(1, 32, 32, 3)
        x_to_select_svhn.append(d1)
        y_to_select_svhn.append(train_y_ori_data[i])
        aug_or_ori.append(0)

    for j in range(7000):  # 12000, select 6000
        d2 = train_x_aug_data[7000 + j].reshape(1, 32, 32, 3)
        x_to_select_svhn.append(d2)
        y_to_select_svhn.append(train_y_aug_data[7000 + j])
        aug_or_ori.append(1)

    x_save_data = np.array(x_to_select_svhn)
    y_save_data = np.array(y_to_select_svhn)
    aug_or_ori = np.array(aug_or_ori)
    state = np.random.get_state()
    np.random.shuffle(x_save_data)
    np.random.set_state(state)
    np.random.shuffle(y_save_data)
    np.random.set_state(state)
    np.random.shuffle(aug_or_ori)

    for k in range(len(test_x_ori_data)):
        dd1 = test_x_ori_data[k].reshape(1, 32, 32, 3)
        x_retrain_mix_test.append(dd1)
        y_retrain_mix_test.append(test_y_ori_data[k])
        x_retrain_ori_test.append(dd1)
        y_retrain_ori_test.append(test_y_ori_data[k])

    for p in range(len(test_x_aug_data)):
        dd2 = test_x_aug_data[p].reshape(1, 32, 32, 3)
        x_retrain_mix_test.append(dd2)
        y_retrain_mix_test.append(test_y_aug_data[p])
        x_retrain_aug_test.append(dd2)
        y_retrain_aug_test.append(test_y_aug_data[p])

    print(len(x_save_data))

    os.makedirs("../../gen_data/svhn_retrain", exist_ok=True)
    np.savez("../../gen_data/svhn_retrain/svhn_toselect", X=x_save_data, Y=y_save_data)
    np.save("../../gen_data/svhn_retrain/data_type", aug_or_ori)
    np.savez("../../gen_data/svhn_retrain/svhn_ori_test", X=x_retrain_ori_test, Y=y_retrain_ori_test)
    np.savez("../../gen_data/svhn_retrain/svhn_aug_test", X=x_retrain_aug_test, Y=y_retrain_aug_test)
    np.savez("../../gen_data/svhn_retrain/svhn_mix_test", X=x_retrain_mix_test, Y=y_retrain_mix_test)


if __name__ == '__main__':
    parse = argparse.ArgumentParser("Generate the to-be-selected dataset for retrain.")
    parse.add_argument('-dataset', required=True, choices=['mnist', 'snips', 'fashion', 'agnews','svhn'])
    args = parse.parse_args()

    if args.dataset == "mnist":
        gen_mnist()

    if args.dataset == "snips":
        gen_snips()

    if args.dataset == "fashion":
        gen_fashion()

    if args.dataset == "agnews":
        gen_agnews()

    if args.dataset == "svhn":
        gen_svhn()
