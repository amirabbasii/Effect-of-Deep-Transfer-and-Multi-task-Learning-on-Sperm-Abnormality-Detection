
import tensorflow as tf
import numpy as np
import modules.LoadData as ld
import statistics


class DMTL_ModelCheckpoint(tf.keras.callbacks.Callback):
        def __init__(self, x_val, y_val, model,name_of_best_weight,label_index):
            self.x_val = x_val
            self.y_val = y_val
            self.label_index=label_index
            self.__model = model
            self.best_acc = -1
            self.minimum_std=10000
            self.best_loss=10000
            self.name_of_best_weight=name_of_best_weight
            self.lastans=[]
            preprocess_config = {
                'shift_range': 5,
                'rotate_range': 360.0,
                'flip_ud': True,
                'flip_lr': True,
                'scale_range': 1.25,
            }
            self.data_aug = ld.DataAug(**preprocess_config)

        def on_epoch_end(self, epoch, logs={}):
            X = self.x_val
            Y = self.y_val
            if self.label_index == -1:
                tmp = self.__model.predict(X)
                tmp = tmp > 0.5
                tmp = tmp * 1
                ans = tmp == Y
                a_acc = np.sum(ans[:, 0]) / len(ans)
                v_acc = np.sum(ans[:, 1]) / len(ans)
                h_acc = np.sum(ans[:, 2]) / len(ans)
                std = [float(a_acc), float(v_acc), float(h_acc), float(4 - (a_acc + v_acc + h_acc))]
                std = statistics.stdev(std)
                print(a_acc, v_acc, h_acc)
                if std < self.minimum_std:
                    self.__model.save_weights(self.name_of_best_weight)
                    self.minimum_std = std
                    print("std updated to ",std)

            else:
                tmp = self.__model.evaluate(X, Y)
                loss=tmp[1:4][self.label_index]
                acc = tmp[4:][self.label_index]
                acc = float("{0:.3f}".format(acc))
                if (acc>self.best_acc) or (self.best_acc==acc and self.best_loss<loss):
                    self.best_acc=acc
                    self.best_loss=loss
                    print("updated to ",str(acc))


class DTL_ModelCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, x_val, y_val, model, name_of_best_weight):
        self.x_val = x_val
        self.y_val = y_val
        self.__model = model
        self.best_acc = 0
        self.best_loss = 10000
        self.name_of_best_weight = name_of_best_weight
        self.lastans = []

    def on_epoch_end(self, epoch, logs={}):
        tmp = self.__model.evaluate(self.x_val, self.y_val, verbose=0)
        loss = tmp[0]
        acc = tmp[1]
        acc = float("{0:.3f}".format(acc))
        if (acc > self.best_acc or (acc == self.best_acc and loss < self.best_loss)):
            self.__model.save_weights(self.name_of_best_weight)

            self.best_acc = acc

            self.best_loss = loss

            print("updated to ", str(loss))


