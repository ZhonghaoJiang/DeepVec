from gensim.models import Word2Vec
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
import re
import os
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Dense, GRU, Embedding, Dropout, Input, Lambda
from keras.models import Model
from keras.callbacks import ModelCheckpoint
import keras
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

intent_dic = {"PlayMusic": 0, "AddToPlaylist": 1, "RateBook": 2, "SearchScreeningEvent": 3,
              "BookRestaurant": 4, "GetWeather": 5, "SearchCreativeWork": 6}


def load_sentence(filename):
    df = pd.read_csv(filename)
    sentences = list(df["text"])
    words = []
    for s in sentences:
        clean = re.sub(r'[^ a-z A-Z 0-9]', " ", s)
        w = word_tokenize(clean)
        # stemming
        words.append([i.lower() for i in w])
    return words


def process_data(data_path, save_path):
    standard_data_save_path = os.path.join(save_path, "standard_data")
    embedding_matrix_save_path = os.path.join(save_path, "embedding_matrix")
    w2v_model_save_path = os.path.join(save_path, "w2v_model")
    words = load_sentence(data_path)

    w2v_model = Word2Vec(sentences=words, size=256, min_count=1, window=5, workers=4)
    embedding_matrix = w2v_model.wv.vectors
    np.save(embedding_matrix_save_path, embedding_matrix)

    # Get all words
    vocab_list = list(w2v_model.wv.vocab.keys())
    # Index corresponding to each word
    word_index = {word: index for index, word in enumerate(vocab_list)}

    # Serialization
    def get_index(sentence):
        sequence = []
        for word in sentence:
            try:
                sequence.append(word_index[word])
            except KeyError:
                pass
        return sequence

    X_data = list(map(get_index, words))

    maxlen = 16
    X_pad = pad_sequences(X_data, maxlen=maxlen)

    # Get the labels
    df = pd.read_csv(data_path)
    intent_ = df["intent"]
    intent = [intent_dic[i] for i in intent_]
    Y = keras.utils.to_categorical(intent, num_classes=7)

    # Split dataset
    # X_train, X_test, Y_train, Y_test = train_test_split(X_pad, Y, test_size=0.2, shuffle=False)
    X_train = X_pad[999:9990]  # 0:999(to select) 999:9990(train) 9990:12488(test)
    X_test = X_pad[9990:]
    Y_train = Y[999:9990]
    Y_test = Y[9990:]

    # Let Keras' Embedding layer use the trained Word2Vec weights
    embedding_matrix = w2v_model.wv.vectors

    # These should be stored in npz to avoid regenerating each time it is executed
    np.savez(standard_data_save_path, X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test)
    np.save(embedding_matrix_save_path, embedding_matrix)
    w2v_model.save(w2v_model_save_path)
    # return (X_train, Y_train), (X_test, Y_test), embedding_matrix


def load_aug_sentence(train_file,test_file):
    df1 = pd.read_csv(train_file)
    df2 = pd.read_csv(test_file)
    df = pd.concat([df1, df2])
    sentences = list(df["text"])
    words = []
    for s in sentences:
        clean = re.sub(r'[^ a-z A-Z 0-9]', " ", str(s))
        w = word_tokenize(clean)
        # stemming
        words.append([i.lower() for i in w])
    return words

def process_aug_data(train_data_path, test_data_path, save_path):
    standard_data_save_path = os.path.join(save_path, "standard_data")
    embedding_matrix_save_path = os.path.join(save_path, "embedding_matrix")
    w2v_model_save_path = os.path.join(save_path, "w2v_model")
    words = load_aug_sentence(train_data_path, test_data_path)

    w2v_model = Word2Vec(sentences=words, size=256, min_count=1, window=5, workers=4)
    embedding_matrix = w2v_model.wv.vectors
    np.save(embedding_matrix_save_path, embedding_matrix)

    # Get all words
    vocab_list = list(w2v_model.wv.vocab.keys())
    # Index corresponding to each word
    word_index = {word: index for index, word in enumerate(vocab_list)}

    # Serialization
    def get_index(sentence):
        sequence = []
        for word in sentence:
            try:
                sequence.append(word_index[word])
            except KeyError:
                pass
        return sequence

    X_data = list(map(get_index, words))

    maxlen = 16
    X_pad = pad_sequences(X_data, maxlen=maxlen)

    # Get the labels
    df1 = pd.read_csv(train_data_path)
    df2 = pd.read_csv(test_data_path)
    df = pd.concat([df1, df2])
    intent_ = df["intent"]
    intent = [intent_dic[i] for i in intent_]
    Y = keras.utils.to_categorical(intent, num_classes=7)

    # Split dataset
    X_train = X_pad[:len(df1)]  # 0:999(to select) 999:9990(train) 9990:12488(test)
    X_test = X_pad[len(df1):]
    Y_train = Y[:len(df1)]
    Y_test = Y[len(df1):]

    # These should be stored in npz to avoid regenerating each time it is executed
    embedding_matrix = w2v_model.wv.vectors

    # These should be stored in npz to avoid regenerating each time it is executed
    np.savez(standard_data_save_path, X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test)
    np.save(embedding_matrix_save_path, embedding_matrix)
    w2v_model.save(w2v_model_save_path)

def load_data(path):
    with np.load(path, allow_pickle=True) as f:
        x_train, y_train = f['X_train'], f['Y_train']
        x_test, y_test = f['X_test'], f['Y_test']
    return (x_train, y_train), (x_test, y_test)


class SnipsGRUClassifier:
    def __init__(self):
        # Classifier
        self.data_path = None
        self.embedding_path = None
        self.vocab_size = None
        self.max_length = 16
        self.padded_doc = None
        self.word_tokenizer = None
        self.embedding_matrix = None
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None
        self.n_units = 64  # hidden LSTM units
        self.n_epochs = 20
        self.batch_size = 2048  # Size of each batch
        self.n_classes = 7

    def get_information(self):
        (self.X_train, self.Y_train), (self.X_test, self.Y_test) = load_data(self.data_path)
        self.embedding_matrix = np.load(self.embedding_path)

    def create_model(self):
        self.get_information()
        input = Input(shape=(self.max_length,))
        embedding = Embedding(
            input_dim=self.embedding_matrix.shape[0],
            output_dim=self.embedding_matrix.shape[1],  # 256
            input_length=self.max_length,
            weights=[self.embedding_matrix],
            mask_zero=True,
            trainable=False, name="embedding")(input)
        gru = GRU(self.n_units, return_sequences=True, dropout=0.5, name='gru')(embedding)
        last_timestep = Lambda(lambda x: x[:, -1, :])(gru)
        dense1 = Dense(64, activation="relu", name='dense1')(last_timestep)
        dropout = Dropout(0.3, name='drop')(dense1)
        dense2 = Dense(self.n_classes, activation="softmax", name='dense2')(dropout)
        self.model = Model(inputs=input, outputs=dense2)
        self.model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        self.model.summary()

    def train_model(self, save_path):
        self.create_model()
        checkpoint = ModelCheckpoint(filepath=os.path.join(save_path, "snips_gru.h5"),
                                     monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
        self.model.fit(self.X_train, self.Y_train, epochs=self.n_epochs, batch_size=self.batch_size,
                       validation_data=(self.X_test, self.Y_test), callbacks=[checkpoint])
        os.makedirs(save_path, exist_ok=True)
        self.model.save(os.path.join(save_path, "snips_gru.h5"))
        print(self.model.evaluate(self.X_test, self.Y_test))

    def train_model_(self, save_path):
        self.create_model()
        checkpoint = ModelCheckpoint(filepath=os.path.join(save_path, "snips_gru_ori.h5"),
                                     monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
        self.model.fit(self.X_train, self.Y_train, epochs=self.n_epochs, batch_size=self.batch_size,
                       validation_data=(self.X_test, self.Y_test), callbacks=[checkpoint])
        os.makedirs(save_path, exist_ok=True)
        self.model.save(os.path.join(save_path, "snips_gru_ori.h5"))
        print(self.model.evaluate(self.X_test, self.Y_test))

    def retrain(self, X_selected, Y_selected, X_val, Y_val, save_path):
        self.create_model()
        Xa_train = np.concatenate([X_selected, self.X_train])
        Ya_train = np.concatenate([Y_selected, self.Y_train])

        checkpoint = ModelCheckpoint(filepath=save_path, monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
        self.model.fit(Xa_train, Ya_train, validation_data=(X_val, Y_val), epochs=self.n_epochs,
                       batch_size=self.batch_size, callbacks=[checkpoint])

        self.model.save(save_path)

    def dau_train(self, save_path):
        self.create_model()

        checkpoint = ModelCheckpoint(filepath=os.path.join(save_path, "snips_gru_aug.h5"),
                                     monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
        self.model.fit(self.X_train, self.Y_train, epochs=self.n_epochs, batch_size=self.batch_size,
                       validation_data=(self.X_test, self.Y_test), callbacks=[checkpoint])
        os.makedirs(save_path, exist_ok=True)
        self.model.save(os.path.join(save_path, "snips_gru_aug.h5"))
        acc = self.model.evaluate(self.X_test, self.Y_test)[1]
        print("dau acc:", acc)
        return acc

    def dau_retrain(self, ori_model_path, X_selected, Y_selected, X_val, Y_val, save_path):
        self.create_model()
        Xa_train = np.concatenate([X_selected, self.X_train])
        Ya_train = np.concatenate([Y_selected, self.Y_train])

        checkpoint = ModelCheckpoint(filepath=save_path, monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
        self.model.fit(Xa_train, Ya_train, validation_data=(X_val, Y_val), epochs=self.n_epochs,
                       batch_size=self.batch_size, shuffle=False, callbacks=[checkpoint])

        self.model.save(save_path)

    def evaluate_retrain(self, retrain_model_path, ori_model_path, x_val, y_val):
        retrain_model = load_model(retrain_model_path)
        ori_model = load_model(ori_model_path)
        retrain_acc = retrain_model.evaluate(x_val, y_val)[1]
        ori_acc = ori_model.evaluate(x_val, y_val)[1]
        print("retrain acc: ", retrain_acc, "ori acc:", ori_acc)
        return retrain_acc, retrain_acc - ori_acc

    def load_hidden_state_model(self, model_path):
        """
        return the rnn model with return_sequence enabled.
        """
        self.get_information()
        input = Input(shape=(self.max_length,))
        input._keras_history[0].supports_masking = True
        embedding = Embedding(
            input_dim=self.embedding_matrix.shape[0],
            output_dim=self.embedding_matrix.shape[1],  # 256
            input_length=self.max_length,
            weights=[self.embedding_matrix],
            mask_zero=True,
            trainable=False, name="embedding")(input)
        gru = GRU(self.n_units, return_sequences=True, dropout=0.5, name='gru')(embedding)
        last_timestep = Lambda(lambda x: x[:, -1, :])(gru)
        dense1 = Dense(64, activation="relu", name='dense1')(last_timestep)
        dropout = Dropout(0.3, name='drop')(dense1)
        dense2 = Dense(self.n_classes, activation="softmax", name='dense2')(dropout)

        model = Model(inputs=input, outputs=[dense2, gru])
        model.load_weights(model_path)
        # print("weights:", model.get_weights())
        return model

    def reload_dense(self, model_path):
        input = Input(shape=((self.n_units),))
        dense1 = Dense(64, activation="relu", name='dense1')(input)
        dropout = Dropout(0.3, name='drop')(dense1)
        dense2 = Dense(self.n_classes, activation="softmax", name='dense2')(dropout)
        model = Model(inputs=input, outputs=dense2)
        model.load_weights(model_path, by_name=True)
        return model

    def profile_train_data(self, model, profile_save_path):
        self.get_information()
        output = model.predict(self.X_train)
        cls = np.argmax(output[0], axis=1)
        cls_label = np.argmax(self.Y_train, axis=1)
        correct_idx = np.where(cls == cls_label)[0]
        os.makedirs(profile_save_path, exist_ok=True)
        states_correct = output[1][correct_idx]
        np.save(os.path.join(profile_save_path, "states_profile.npy"), states_correct)

    def get_state_profile(self, inputs, model):
        output = model.predict(inputs)
        return output[1]


def train_model():
    # step 1. preprocess data
    data_path = "./data/new_intent.csv"
    save_path = "./save"
    process_data(data_path, save_path)

    # step 2. create and train the model
    classifier = SnipsGRUClassifier()
    classifier.embedding_path = "./save/embedding_matrix.npy"
    classifier.data_path = "./save/standard_data.npz"
    classifier.get_information()
    classifier.train_model("./models")


def train_model_ori():
    # # step 1. preprocess data
    # data_path = "./data/new_intent.csv"
    # save_path = "./save"
    # process_data(data_path, save_path)

    # step 2. create and train the model
    classifier = SnipsGRUClassifier()
    classifier.embedding_path = "./save/embedding_matrix.npy"
    classifier.data_path = "./save/standard_data.npz"
    classifier.get_information()
    classifier.train_model_("./models")

def train_model_dau():
    save_path = "./save_aug"
    base_path = "../../gen_data/gen_train_dau/dau/snips_harder/"
    process_aug_data(f"{base_path}snips_train_aug.csv", f"{base_path}snips_test_aug.csv", save_path)

    classifier = SnipsGRUClassifier()
    classifier.embedding_path = "./save_aug/embedding_matrix.npy"
    classifier.data_path = "./save_aug/standard_data.npz"
    classifier.get_information()
    classifier.dau_train("./models")


if __name__ == "__main__":
    parse = argparse.ArgumentParser("Train the GRU model on Snips dataset.")
    parse.add_argument('-type', required=True, choices=['train', 'retrain', 'dau_train'])
    args = parse.parse_args()

    if args.type == "train":
        train_model()
    elif args.type == "retrain":
        train_model_ori()
    elif args.type == "dau_train":
        train_model_dau()