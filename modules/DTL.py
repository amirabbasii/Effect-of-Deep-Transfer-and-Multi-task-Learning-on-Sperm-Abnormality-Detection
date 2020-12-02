import math
from builtins import enumerate

from sklearn.metrics import confusion_matrix,roc_auc_score
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Lambda
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import save_model, load_model
import random
from imblearn.keras import BalancedBatchGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator as IDG
from sklearn.metrics import confusion_matrix
from tensorboard.plugins.hparams import api as hp
from imblearn.keras import BalancedBatchGenerator, balanced_batch_generator
from scipy.ndimage.filters import median_filter
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ReduceLROnPlateau, LearningRateScheduler, TensorBoard
from tensorflow.keras.layers import concatenate, Dense, Dropout, Input, Activation, Flatten, Conv2D, MaxPooling2D, \
    AveragePooling2D, GlobalAveragePooling2D, BatchNormalization
from modules.Sampler import Sampler
from modules import LoadData as ld
from modules.LoadData import load_data
from modules.Generators import *
from modules.CallBacks import *
import os
import pickle

class DTL():
    def __init__(self, params,base_model,label,data=None):
        default_params = {"agumentation": False, "scale": False, "dense_activation": "relu", "regularizition": 0.0
            , "dropout": 0.0, "optimizer": "adam", "number_of_dense": 1, "balancer": "None", "batch_size": 32}
        default_params.update(params)
        Model = base_model
        params = default_params
        if data==None:
          data = load_data(label=label, phase="search")
        self.batch_size = params["batch_size"]
        if params['agumentation']:
            data["x_val"] = ld.normalize(data["x_val"])
            data["x_test"] = ld.normalize(data["x_test"])
        elif params["scale"]:
            data["x_val"] = ld.normalize(data["x_val"])
            data["x_test"] = ld.normalize(data["x_test"])
            data["x_train"] = ld.normalize(data["x_train"])
        regularization = not (params["regularizition"] == 0.0)

        dropout = not (params["dropout"] == 0.0)

        self.agumnetation = params["agumentation"]

        ############ Creating CNN ##############
        optimizer = params["optimizer"]
        inp = Input((128,128, 1))
        con = concatenate([inp, inp, inp])
        model = Model(include_top=False, weights='imagenet', input_tensor=con)
        x = Flatten()(model.layers[-1].output)

        for i in range(params["number_of_dense"]):
            if regularization:
                x = Dense(params["nn"], activation=params["dense_activation"],
                          kernel_regularizer=l2(params["regularizition"]))(x)
            else:
                x = Dense(params["nn"], activation=params["dense_activation"])(x)
            if dropout:
                x = Dropout(params["dropout"])(x)
        x = Dense(1, activation="sigmoid", name="classification")(x)
        model = tf.keras.Model(model.input, x)
        model.compile(optimizer=optimizer, metrics=["accuracy"], loss=params["loss"])

        self.__model = model
        self.__data = data
        
        self.balancer = params["balancer"]
        self.__number_of_dense = params["number_of_dense"]
        self.details = [list(params.keys())[i] + ":" + str(list(params.values())[i]) for i in range(len(params))]

    def train(self, epochs, load_best_weigth, verbose, TensorB, name_of_best_weight, phase):
        if phase == "train" and self.agumnetation:
            self.__data["x_train"] = self.__data["x_train_128"]
        # self.__data["x_val"] = self.__data["x_val_128"]
        batch_size = self.batch_size
        balancer = self.balancer
        callbacks = [DTL_ModelCheckpoint(self.__data["x_val"], self.__data["y_val"], self.__model, name_of_best_weight)]
        if TensorB:  # save History of train
            tb = TensorBoard(log_dir="log_train/" + "#".join(self.details))
            callbacks.append(tb)
        '''
        Here we have two kinds of sampler.
        online,offline
        online:for fix data on every epoch(we will use it in fit_generator)
        offline:for fix data before train

        onlines 1-batch_balancer:we can just use Data Agumentation with it
                2-DySa:nothing can't combine with it
        offlines:if balamcer be one of {smote,adasyn,None},then we should just our balancer to fix data and then use agumentation or not.

        '''
        ##batch_balancer
        if balancer == "batch_balancer":
            S = [[], []]
            for i in range(len(self.__data["x_train"])):
                S[self.__data["y_train"][i][0]].append(self.__data["x_train"][i])
            S = np.array(S)
            generator = BatchBalancer(S, self.agumnetation, batch_size)
            hist = self.__model.fit_generator(generator, validation_data=(self.__data["x_val"], self.__data["y_val"]),
                                              shuffle=True, callbacks=callbacks, steps_per_epoch=1000 / batch_size,
                                              epochs=epochs, verbose=verbose)
        else:  ###It means that we will use offline balancers
            self.__data = self.__getBalancedData(balancer)  # fixing data
            if self.agumnetation:
                generator = Agumentation(self.__data, batch_size)
                hist = self.__model.fit_generator(generator,
                                                  validation_data=(self.__data["x_val"], self.__data["y_val"]),
                                                  shuffle=True, callbacks=callbacks, steps_per_epoch=1000 / batch_size,
                                                  epochs=epochs, verbose=verbose)
            ###It means that we will use offline balancers
            else:

                hist = self.__model.fit(self.__data["x_train"], self.__data["y_train"], epochs=epochs,
                                        batch_size=batch_size,
                                        validation_data=(self.__data["x_val"], self.__data["y_val"]), shuffle=True,
                                        callbacks=callbacks, verbose=verbose)
        if load_best_weigth:
            self.__model.load_weights(name_of_best_weight)
        save_model(self.__model, "model_" + name_of_best_weight)

        # cleaning model from GPU

    def clear(self):
        tf.keras.backend.clear_session()
        del self.__model
        del self.__data

    def __getBalancedData(self, name):
        if name == "None":  # no sampler
            return self.__data
        elif name == "smote":
            return Sampler("smote", 10, self.__data).run()
        elif name == "adasyn":
            return Sampler("adasyn", 10, self.__data).run()

    def evaluate(self):
        X = self.__data["x_test"]
        y = self.__data["y_test"]
        y_pred1 = self.__model.predict(X)
        y_pred = y_pred1 > 0.5
        y_pred = y_pred * 1
        c_matrix = confusion_matrix(y, y_pred)
        precision = c_matrix[0, 0] / sum(c_matrix[:,0])
        recall = c_matrix[0, 0] / sum(c_matrix[ 0])
        acc = np.sum(c_matrix.diagonal()) / np.sum(c_matrix)
        f_half = 1.25 * precision * recall / (.25 * precision + recall)
        g_mean = math.sqrt(precision * recall)
        TP = c_matrix[0][0]
        TN = c_matrix[1][1]
        FN= c_matrix[0][1]
        FP= c_matrix[1][0]
        mcc = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
        auc = roc_auc_score(y, y_pred)
        return [acc,precision,recall,f_half,g_mean,auc,mcc]
    @staticmethod
    def k_fold(k,label, epochs, params, load_best_weigth, verbose, TensorB, name_of_best_weight,base_model):
        flag = params['agumentation']
        data = ld.load_data(label, phase="aug_evaluation") if flag==True else ld.load_data(label, phase="evaluation")
        results=[]
        size = len(data['x']) // k
        tmp_idx = np.arange(data['x'].shape[0])
        np.random.shuffle(tmp_idx)
        x = data['x'][tmp_idx]
        y = data['y'][tmp_idx]
        np.save("x.npy", x)
        np.save("y.npy", y)
        acc_vec = []
        for i in range(k):
            x_test = x[i * size:(i + 1) * size]
            y_test = y[i * size:(i + 1) * size]
            x_train = np.append(x[0:i * size], x[(i + 1) * size:], axis=0)
            y_train = np.append(y[0:i * size], y[(i + 1) * size:], axis=0)
            tmp = random.sample(range(len(x_train)), 232)
            x_val = []
            y_val = []
            for j in tmp:
                x_val.append(x_train[j])
                y_val.append(y_train[j])
            x_val = np.array(x_val)
            y_val = np.array(y_val)
            x_train = np.delete(x_train, tmp, axis=0)
            y_train = np.delete(y_train, tmp, axis=0)

            ##########fixing data#########
            data = ld.fix_data(flag, x_train, y_train, x_val, y_val, x_test, y_test)

            model = DTL(params=params,base_model=base_model,label=label,data=data)
            model.train(epochs, load_best_weigth, verbose, TensorB, name_of_best_weight + str(i) + ".h5", "k_fold")
            results.append(model.evaluate())
            print(results[-1])
            model.clear()
            del model
        return results





