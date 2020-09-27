
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


class DMTL:
    def stage_model(self,stage,second_model,params,loss,base_model):
      Model=base_model
      optimizer = params["optimizer"]
      if stage==2:
          metrics = ["accuracy"]
          labels = ['h','v','a']
          label_index=self.label_index
          for l in second_model.layers:
            l.trainable = False
          y=second_model.layers[-2].output
          out=[]
          for k in range(len(labels)):
              x=y
              for i in range(params["number_of_dense"][k]):
                  if params["regularizition"][k] != 0.0:
                      x = Dense(params["nn"][k], activation=params["dense_activation"][k],trainable=(self.label_index==k),
                                kernel_regularizer=l2(params["regularizition"][k]), name="dense_" + labels[k] + str(i))(x)
                  else:
                      x = Dense(params["nn"][k], activation=params["dense_activation"][k],trainable=(self.label_index==k),
                                name="dense_" + labels[k] + str(i))(x)
                  if params["dropout"][k] != 0.0:
                      x = Dropout(params["dropout"][k],trainable=(self.label_index==k), name="dropout_" + labels[k] + str(i))(x)
              x = Dense(1, activation="sigmoid",trainable=(self.label_index==k),name="class_"+labels[k])(x)
              out.append(x)
          model = tf.keras.Model(second_model.input,out)
          model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
          return model
      else:
          metrics = [Meric.a, Meric.v, Meric.h, Meric.loss_a, Meric.loss_v, Meric.loss_h]

          inp = Input((64, 64, 1))
          con = concatenate([inp, inp, inp])

          model = Model(include_top=False, weights='imagenet', input_tensor=con)
          
          y=Flatten()(model.layers[-1].output)
          for i in range(params["number_of_dense"]):
              x=y
              if params["regularizition"]!=0.0:
                  
                  x = Dense(params["nn"], activation=params["dense_activation"],kernel_regularizer=l2(params["regularizition"]), name="BigDense")(x)
              else:
                  
                  x = Dense(params["nn"], activation=params["dense_activation"],name="BigDense")(x)
          if params["dropout"]!=0.0:
            x = Dropout(params["dropout"], name="BigDrop")(x)
          x = Dense(3, activation="sigmoid",name="classification")(x)
          model = tf.keras.Model(inp,x)
          model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
          return model
    def __init__(self,params, base_model, label, loss, second_model,phase,data=None):
        if data==None:
          data=self.prepare_data(phase,params)
        self.phase=phase
        labels = ['h','v','a']
        self.label_index = labels.index(label) if phase==2 else -1
        default_params = {"agumentation": False, "scale": False, "dense_activation": "relu", "regularizition": 0.0
            , "dropout": 0.0, "optimizer": "adam", "number_of_dense": 1, "balancer": "None", "batch_size": 32,
                          "nn": 1024}
        default_params.update(params)
        
        params = default_params

        self.agumnetation = params["agumentation"]

        ############ Creating CNN ##############
        optimizer = params["optimizer"]

        model=self.stage_model(phase,second_model,params,loss,base_model)
        self.__model = model
        self.__data = data
        self.balancer = params["balancer"]
        self.batch_size=params["batch_size"]
        self.__number_of_dense = params["number_of_dense"]
        tmp = list(params.values())
        keys = list(params.keys())
        if phase==2:
          for i in range(len(keys)):
              if keys[i] in ["dense_activation", "regularizition", "dropout", "number_of_dense"]:
                  tmp[i] = tmp[i][self.label_index]

        self.details = [keys[i] + ":" + str(tmp[i]) for i in range(len(keys))]
    def train(self, epochs, load_best_weigth, verbose, TensorB, name_of_best_weight, phase,save_m):
        if phase == "train" and self.agumnetation:
            self.__data["x_train"] = self.__data["x_train_128"]
        # self.__data["x_val"] = self.__data["x_val_128"]
        batch_size = self.batch_size
        balancer = self.balancer
        callbacks = [DMTL_ModelCheckpoint(self.__data["x_val"], self.__data["y_val"], self.__model, name_of_best_weight,self.label_index)]
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
        if save_m:
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

    def prepare_data(self,phase,params):
        d_a = load_data(label="a", phase="search")
        d_h = load_data(label="h", phase="search")
        d_v = load_data(label="v", phase="search")
        y = {"y_val", "y_train", "y_test"}
        x = {"x_val", "x_train", "x_test", "x_val_128", "x_train_128"}
        data = {}
        for t in y:
            if phase == 1:
                data[t] = np.array([[d_a[t][i][0], d_v[t][i][0], d_h[t][i][0]] for i in range(len(d_a[t]))])
            else:
                data[t] = [d_a[t], d_v[t], d_h[t]]
        for t in x:
            data[t] = d_a[t]
        if params['agumentation']:
            data["x_val"] = ld.normalize(data["x_val"])
            data["x_test"] = ld.normalize(data["x_test"])
        elif params["scale"]:
            data["x_val"] = ld.normalize(data["x_val"])
            data["x_test"] = ld.normalize(data["x_test"])
            data["x_train"] = ld.normalize(data["x_train"])
        return data

    def evaluate(self):

        X = self.__data["x_test"]
        y = self.__data["y_test"][:,self.label_index] if self.phase==1 else self.__data["y_test"][self.label_index]
        y_pred1 = self.__model.predict(X)
        y_pred = y_pred1[:,self.label_index] > 0.5 if self.phase==1 else y_pred1[self.label_index] > 0.5
        y_pred = y_pred * 1
        c_matrix = confusion_matrix(y, y_pred)
        precision = c_matrix[0, 0] / sum(c_matrix[:,0])
        recall = c_matrix[0, 0] / sum(c_matrix[ 0])
        acc = np.sum(c_matrix.diagonal()) / np.sum(c_matrix)
        f_half = 1.25 * precision * recall / (.25 * precision + recall)
        g_mean = math.sqrt(precision * recall)
        TP = c_matrix[0][0]
        TN = c_matrix[1][1]
        FN = c_matrix[0][1]
        FP= c_matrix[1][0]
        mcc = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
        auc = roc_auc_score(y, y_pred)
        return [acc, precision, recall, f_half, g_mean, auc, mcc]

    @staticmethod
    def k_fold(k,epochs, params1,params2, load_best_weigth, verbose, TensorB, name_of_best_weight,base_model):
        data_tmp = [None, None, None]
        flag = None
        if params2['agumentation']:
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
        np.save("x.npy",x)
        np.save("y.npy",y)
        for i in range(k):
            x_test = x[i * size:(i + 1) * size]

            y_test = [y[0][i * size:(i + 1) * size], y[1][i * size:(i + 1) * size], y[2][i * size:(i + 1) * size]]
            x_train = np.append(x[0:i * size], x[(i + 1) * size:], axis=0)
            y_train = [np.append(y[0][0:i * size], y[0][(i + 1) * size:], axis=0),
                       np.append(y[1][0:i * size], y[1][(i + 1) * size:], axis=0),
                       np.append(y[2][0:i * size], y[2][(i + 1) * size:], axis=0)]

            y_test1 = np.array([[y[0][i][0],y[1][i][0],y[2][i][0]] for i in range(i * size,(i + 1) * size)])
            y_train1=[[y[0][i][0], y[1][i][0], y[2][i][0]] for i in range(i * size)]+[[y[0][i][0], y[1][i][0], y[2][i][0]] for i in range((i + 1) * size,len( y[2]))]
            y_train1=np.array(y_train1)
            tmp = random.sample(range(len(x_train)), 232)
            x_val = []
            y_val = [[], [], []]
            y_val1=[]
            for j in tmp:
                x_val.append(x_train[j])
                y_val[0].append(y_train[0][j])
                y_val[1].append(y_train[1][j])
                y_val[2].append(y_train[2][j])
                y_val1.append([y_train[0][j][0],y_train[1][j][0],y_train[2][j][0]])
            y_val1=np.array(y_val1)
            x_val = np.array(x_val)

            x_train = np.delete(x_train, tmp, axis=0)
            y_train[0] = np.delete(y_train[0], tmp, axis=0)
            y_train[1] = np.delete(y_train[1], tmp, axis=0)
            y_train[2] = np.delete(y_train[2], tmp, axis=0)
            y_train1=np.delete(y_train1, tmp, axis=0)
            for j in range(3):
                y_val[j]=np.array(y_val[j])
            ##########fixing data#########
            data2 = ld.fix_data(flag, x_train, y_train, x_val, y_val, x_test, y_test)
            data1 = ld.fix_data(flag, x_train, y_train1, x_val, y_val1, x_test, y_test1)

            model = DMTL(params=params1,base_model=base_model,label="",loss='binary_crossentropy',second_model=None,phase=1,data=data1)

            model.train(1, False, verbose, TensorB, name_of_best_weight + str(i) + ".h5", "k_fold",save_m=True)
            
            epochs=[1,1,1]
            labels=['a','v','h']
            for l in range(3):
                second_model = load_model("model_"+name_of_best_weight + str(i) + ".h5",custom_objects={
                  "a":Meric.a,"v":Meric.v,"h":Meric.h,"loss_a":Meric.loss_a,"loss_v":Meric.loss_v,"loss_h":Meric.loss_h
                })
            
                model = DMTL(params=params2,base_model=base_model,label=labels[l],loss='binary_crossentropy',second_model=second_model,phase=2,data=data2)
                model.train(epochs[l], load_best_weigth, verbose, TensorB, name_of_best_weight + str(i) + ".h5", "k_fold",save_m=False)
                ans =model.evaluate()
                acc_vec.append(ans)
                print(ans)
                model.clear()
                del model
        return acc_vec


class Meric:
    def loss_a(y_true, y_pred):
        a_pred = y_pred[:, 0]
        a_true = y_true[:, 0]
        bce = tf.keras.losses.BinaryCrossentropy()
        return bce(a_true, a_pred)

    def loss_v(y_true, y_pred):
        a_pred = y_pred[:, 1]
        a_true = y_true[:, 1]
        bce = tf.keras.losses.BinaryCrossentropy()
        return bce(a_true, a_pred)

    def loss_h(y_true, y_pred):
        a_pred = y_pred[:, 2]
        a_true = y_true[:, 2]
        bce = tf.keras.losses.BinaryCrossentropy()
        return bce(a_true, a_pred)

    def a(y_true, y_pred):
        a_pred = y_pred[:, 0]
        a_true = y_true[:, 0]
        bce = tf.keras.losses.BinaryCrossentropy()
        return tf.keras.metrics.binary_accuracy(a_true, a_pred)

    def v(y_true, y_pred):
        a_pred = y_pred[:, 1]
        a_true = y_true[:, 1]
        bce = tf.keras.losses.BinaryCrossentropy()
        return tf.keras.metrics.binary_accuracy(a_true, a_pred)

    def h(y_true, y_pred):
        a_pred = y_pred[:, 2]
        a_true = y_true[:, 2]
        bce = tf.keras.losses.BinaryCrossentropy()
        return tf.keras.metrics.binary_accuracy(a_true, a_pred)


