import keras
import os
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Input, Lambda, LSTM, Dense, Bidirectional, Dropout
from keras.models import load_model
from keras.models import Model
import numpy as np
import os
from keras.callbacks import ModelCheckpoint
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class MnistBLSTMClassifier:
    def __init__(self):
        # Classifier
        self.time_steps = 28  # timesteps to unroll
        self.n_units = 128  # hidden LSTM units
        self.n_inputs = 28  # rows of 28 pixels (an mnist img is 28x28)
        self.n_classes = 10  # mnist classes/labels (0-9)
        self.batch_size = 1024  # Size of each batch
        self.n_epochs = 30
        self.epochs = 20

    def input_preprocess(self, data):
        data = data.reshape(data.shape[0], self.n_inputs, self.n_inputs)
        data = data.astype('float32')
        data /= 255
        return data

    def create_model(self):
        input = Input(shape=(self.time_steps, self.n_inputs))
        lstm = Bidirectional(
            LSTM(self.n_units, input_shape=(self.time_steps, self.n_inputs), return_sequences=True, name='lstm'))(input)
        last_timestep = Lambda(lambda x: x[:, -1, :])(lstm)
        dense1 = Dense(64, activation="relu", name='dense1')(last_timestep)
        dropout = Dropout(0.4, name='drop')(dense1)
        dense2 = Dense(self.n_classes, activation="softmax", name='dense2')(dropout)
        self.model = Model(inputs=input, outputs=dense2)
        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        # self.model.summary()

    def train(self, save_path):
        self.create_model()
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        x_train = self.input_preprocess(x_train[:-6000])
        x_test = self.input_preprocess(x_test)

        y_test = keras.utils.to_categorical(y_test, num_classes=10)
        y_train = keras.utils.to_categorical(y_train[:-6000], num_classes=10)

        checkpoint = ModelCheckpoint(filepath=os.path.join(save_path, "mnist_blstm.h5"), monitor='val_acc', mode='auto',
                                     save_best_only='True')
        self.model.fit(x_train, y_train, validation_data=(x_test, y_test),
                       batch_size=self.batch_size, epochs=self.epochs, shuffle=False, callbacks=[checkpoint])

        os.makedirs(save_path, exist_ok=True)
        self.model.save(os.path.join(save_path, "mnist_blstm.h5"))

    def train_(self, save_path):
        self.create_model()
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        x_train = self.input_preprocess(x_train[:-6000])
        x_test = self.input_preprocess(x_test)

        y_test = keras.utils.to_categorical(y_test, num_classes=10)
        y_train = keras.utils.to_categorical(y_train[:-6000], num_classes=10)

        checkpoint = ModelCheckpoint(filepath=os.path.join(save_path, "mnist_blstm_ori.h5"), monitor='val_acc', mode='auto',
                                     save_best_only='True')
        self.model.fit(x_train, y_train, validation_data=(x_test, y_test),
                       batch_size=self.batch_size, epochs=self.n_epochs, shuffle=False, callbacks=[checkpoint])

        os.makedirs(save_path, exist_ok=True)
        self.model.save(os.path.join(save_path, "mnist_blstm_ori.h5"))

    def dau_train(self, save_path, x_dau_train, x_dau_test, y_dau_train, y_dau_test):
        self.create_model()
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_dau_train = np.load(x_dau_train)
        x_dau_test = np.load(x_dau_test)
        y_dau_train = np.load(y_dau_train)
        y_dau_test = np.load(y_dau_test)

        x_train = np.concatenate([x_dau_train, x_train[:-6000]])
        y_train = np.concatenate([y_dau_train, y_train[:-6000]])

        x_train = self.input_preprocess(x_train)
        x_test = self.input_preprocess(x_test)

        y_test = keras.utils.to_categorical(y_test, num_classes=10)
        y_train = keras.utils.to_categorical(y_train, num_classes=10)

        checkpoint = ModelCheckpoint(filepath=os.path.join(save_path, "mnist_blstm_dau.h5"), monitor='val_acc',
                                     mode='auto',
                                     save_best_only=True)
        self.model.fit(x_train, y_train, validation_data=(x_test, y_test),
                       batch_size=self.batch_size, epochs=self.n_epochs, shuffle=False, callbacks=[checkpoint])

        os.makedirs(save_path, exist_ok=True)
        self.model.save(os.path.join(save_path, "mnist_blstm_dau.h5"))

        x_val = self.input_preprocess(x_dau_test)
        y_val = keras.utils.to_categorical(y_dau_test, num_classes=10)
        acc = self.model.evaluate(x_val, y_val)[1]
        print("dau acc:", acc)
        return acc

    def retrain(self, X_selected, Y_selected, X_val, Y_val, save_path):
        self.create_model()
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        Xa_train = np.concatenate([X_selected, x_train[:-6000]])
        Ya_train = np.concatenate([Y_selected, y_train[:-6000]])

        Xa_train = self.input_preprocess(Xa_train)
        X_val = self.input_preprocess(X_val)
        Ya_train = keras.utils.to_categorical(Ya_train, num_classes=10)
        Y_val = keras.utils.to_categorical(Y_val, num_classes=10)

        checkpoint = ModelCheckpoint(filepath=save_path, monitor='val_acc', mode='auto',
                                     save_best_only='True')
        self.model.fit(Xa_train, Ya_train, validation_data=(X_val, Y_val),
                       batch_size=self.batch_size, epochs=self.n_epochs, shuffle=False, callbacks=[checkpoint])

        self.model.save(save_path)

    def evaluate_retrain(self, retrain_model_path, ori_model_path, x_val, y_val):
        x_val = self.input_preprocess(x_val)
        y_val = keras.utils.to_categorical(y_val, num_classes=10)

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
        input = Input(shape=(self.time_steps, self.n_inputs))
        lstm = Bidirectional(
            LSTM(self.n_units, input_shape=(self.time_steps, self.n_inputs), return_sequences=True, name='lstm'))(input)
        last_timestep = Lambda(lambda x: x[:, -1, :])(lstm)
        dense1 = Dense(64, activation="relu", name='dense1')(last_timestep)
        dropout = Dropout(0.4, name='drop')(dense1)
        dense2 = Dense(self.n_classes, activation="softmax", name='dense2')(dropout)

        model = Model(inputs=input, outputs=[dense2, lstm])
        model.load_weights(model_path)
        # self.model = model
        return model

    def reload_dense(self, model_path):
        input = Input(shape=((self.n_units * 2),))
        dense1 = Dense(64, activation="relu", name='dense1')(input)
        dropout = Dropout(0.4, name='drop')(dense1)
        dense2 = Dense(self.n_classes, activation="softmax", name='dense2')(dropout)
        model = Model(inputs=input, outputs=dense2)
        model.load_weights(model_path, by_name=True)
        return model

    def profile_train_data(self, model, profile_save_path):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = self.input_preprocess(x_train)
        output = model.predict(x_train)
        cls = np.argmax(output[0], axis=1)
        correct_idx = np.where(cls == y_train)[0]
        os.makedirs(profile_save_path, exist_ok=True)
        states_correct = output[1][correct_idx]
        np.save(os.path.join(profile_save_path, "states_profile.npy"), states_correct)

    def get_state_profile(self, inputs, model):
        inputs = self.input_preprocess(inputs)
        output = model.predict(inputs)
        return output[1]


def train_model():
    save_path = "./models"
    lstm_classifier = MnistBLSTMClassifier()
    # train an rnn model
    lstm_classifier.create_model()
    lstm_classifier.train(save_path)


def train_model_ori():
    save_path = "./models"
    lstm_classifier = MnistBLSTMClassifier()
    # train an rnn model
    lstm_classifier.create_model()
    lstm_classifier.train_(save_path)

def train_model_dau():
    save_path = "./models"
    lstm_classifier = MnistBLSTMClassifier()
    # train an rnn model
    lstm_classifier.create_model()
    base_path = "./gen_data/gen_train_dau/dau/mnist_harder/"
    lstm_classifier.dau_train(save_path, f"{base_path}x_train_aug.npy", f"{base_path}x_test_aug.npy",
                              f"{base_path}y_train_aug.npy", f"{base_path}y_test_aug.npy")


if __name__ == "__main__":
    parse = argparse.ArgumentParser("Train the BLSTM model on MNIST dataset.")
    parse.add_argument('-type', required=True, choices=['train', 'retrain', 'dau_train'])
    args = parse.parse_args()

    if args.type == "train":
        train_model()
    elif args.type == "retrain":
        train_model_ori()
    elif args.type == "dau_train":
        train_model_dau()