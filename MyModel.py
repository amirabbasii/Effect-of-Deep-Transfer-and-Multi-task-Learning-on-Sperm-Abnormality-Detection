import math
from builtins import enumerate
from DMTL import DMTL
from DTL import DTL
from sklearn.metrics import confusion_matrix
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Lambda
import tensorflow as tf
import numpy as np
from keras.models import save_model, load_model
import random
from modules.mmoe import MMoE
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


class MyModel():
    def __init__(self,type):
        if type=="dmtl":
            self.__model=DMTL()
    def load_model(self, path):
        self.__model = load_model(path)

    def load_weights(self, path):
        self.__model.load_weights(path)

    def get_layers(self):
        return self.__model.layers

    def get_model(self):
        return self.__model
    def train(self, epochs, load_best_weigth, verbose, TensorB, name_of_best_weight, phase):
        if phase == "train" and self.agumnetation:
            self.__data["x_train"] = self.__data["x_train_128"]
        # self.__data["x_val"] = self.__data["x_val_128"]
        batch_size = self.batch_size
        balancer = self.balancer
        callbacks = [self.__checkpoint]
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

    ###evaluate
    ##if result==True we just need accuracy and it will be returned


    # gives balanced data by selected sampler
    # name:name of sampler
    # if name=="None" return data without balancing
    def __getBalancedData(self, name):
        if name == "None":  # no sampler
            return self.__data
        elif name == "smote":
            return Sampler("smote", 10, self.__data).run()
        elif name == "adasyn":
            return Sampler("adasyn", 10, self.__data).run()

    @staticmethod
    def k_fold_train(epochs, k, params, name, load_best_weigth, verbose, TensorB, name_of_best_weight, base_model,
                     label_index, saver, start_unfreeze):
        data_tmp = [None, None, None]
        flag = None
        if params['agumentation']:
            data_tmp[0] = ld.load_data('a', phase="aug_evaluation")
            data_tmp[1] = ld.load_data('v', phase="aug_evaluation")
            data_tmp[2] = ld.load_data('h', phase="aug_evaluation")
            flag = True
        else:
            data_tmp[0] = ld.load_data('a', phase="evaluation")
            data_tmp[1] = ld.load_data('v', phase="evaluation")
            data_tmp[2] = ld.load_data('h', phase="evaluation")
            flag = False
        data = {}
        data['x'] = data_tmp[0]['x']
        data['y'] = [data_tmp[0]['y'], data_tmp[1]['y'], data_tmp[2]['y']]
        size = len(data['x']) // k
        tmp_idx = np.arange(data['x'].shape[0])
        np.random.shuffle(tmp_idx)
        x = data['x'][tmp_idx]
        y = [data['y'][0][tmp_idx], data['y'][1][tmp_idx], data['y'][2][tmp_idx]]
        acc_vec = []
        for i in range(k):
            x_test = x[i * size:(i + 1) * size]
            y_test = [y[0][i * size:(i + 1) * size], y[1][i * size:(i + 1) * size], y[2][i * size:(i + 1) * size]]
            x_train = np.append(x[0:i * size], x[(i + 1) * size:], axis=0)
            y_train = [np.append(y[0][0:i * size], y[0][(i + 1) * size:], axis=0),
                       np.append(y[1][0:i * size], y[1][(i + 1) * size:], axis=0),
                       np.append(y[2][0:i * size], y[2][(i + 1) * size:], axis=0)]
            tmp = random.sample(range(len(x_train)), 232)
            x_val = []
            y_val = [[], [], []]
            for j in tmp:
                x_val.append(x_train[j])
                y_val[0].append(y_train[0][j])
                y_val[1].append(y_train[1][j])
                y_val[2].append(y_train[2][j])
            x_val = np.array(x_val)
            x_train = np.delete(x_train, tmp, axis=0)
            y_train[0] = np.delete(y_train[0], tmp, axis=0)
            y_train[1] = np.delete(y_train[1], tmp, axis=0)
            y_train[2] = np.delete(y_train[2], tmp, axis=0)
            ##########fixing data#########
            data = ld.fix_data(flag, x_train, y_train, x_val, y_val, x_test, y_test)
            model = MyModel(params, data, name, base_model=base_model,
                            label_index=label_index)
            model.train(epochs, load_best_weigth, verbose, TensorB, name_of_best_weight + str(i) + ".h5", "k_fold",
                        None)
            ans = model.get_model().evaluate(data["x_test"], data["y_test"])
            acc = ans[4:]
            acc_vec.append(acc)
            print(acc)
            model.clear()
            del model
        return acc_vec
