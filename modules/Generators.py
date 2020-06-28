import tensorflow as tf
import modules.LoadData as ld
import numpy as np
class BatchBalancer(tf.keras.utils.Sequence):
    def __init__(self, S, agumentation, batch_size, list_IDs=None, labels=None, number_of_classes=2, shuffle=True):
        self.batch_size = batch_size
        self.S = S
        self.agumentation = agumentation
        preprocess_config = {
            'shift_range': 5,
            'rotate_range': 360.0,
            'flip_ud': True,
            'flip_lr': True,
            'scale_range': 1.25,
        }
        self.data_aug = ld.DataAug(**preprocess_config)

    def __getitem__(self, item):
        i = [0, 0]
        X = []
        Y = []
        for j in range(self.batch_size):
            index = np.random.randint(0, 2)
            if i[index] == 0:
                np.random.shuffle(self.S[index])
            img = self.data_aug.augmentation(self.S[index][i[index]]).reshape((64, 64, 1)) if self.agumentation else \
            self.S[index][i[index]]
            X.append(img)
            Y.append(index)

            i[index] = (i[index] + 1) % len(self.S[index])
        return np.array(X), np.array(Y)

    def __len__(self):
        return 1000 // self.batch_size


class Agumentation(tf.keras.utils.Sequence):
    def __init__(self, data, batch_size, list_IDs=None, labels=None, number_of_classes=2, shuffle=True):
        self.batch_size = batch_size
        self.__data = data
        preprocess_config = {
            'shift_range': 5,
            'rotate_range': 360.0,
            'flip_ud': True,
            'flip_lr': True,
            'scale_range': 1.25,
        }
        self.data_aug = ld.DataAug(**preprocess_config)

    def __getitem__(self, item):
        X = []
        Y=[]
        for j in range(self.batch_size):
            if item * self.batch_size + j >= len(self.__data["x_train"]):
                break
            agumented_img = self.data_aug.augmentation(
                self.__data["x_train"][item * self.batch_size + j]).reshape((64, 64, 1))
            X.append(agumented_img)
            Y.append(self.__data["y_train"][item*self.batch_size + j])
        # tmp=self.__data["y_val"]

        # Y=[tmp[0][item * self.batch_size :item * self.batch_size+j],tmp[1][item * self.batch_size :item * self.batch_size+j],tmp[2][item * self.batch_size :item * self.batch_size+j]]
        return np.array(X), np.array(Y)

    def __len__(self):
        return 1000 // self.batch_size
