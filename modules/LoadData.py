import random
import math
import os


import numpy as np
import cv2 as cv


class DataAug:
  def __init__(self, rotate_range, shift_range,
             flip_ud, flip_lr, scale_range):

    # self.canvas_center = canvas_center
    # self.patch_center = patch_center
    self.rotate_range = rotate_range
    self.shift_range = shift_range
    self.flip_ud = flip_ud
    self.flip_lr = flip_lr
    self.scale_log = math.log(scale_range)
    # self.patch_size = patch_size
    self.__initialize_values()

  def __initialize_values(self):
    '''
    Initialized values in this function should be replaced in newer versions
    '''
    image_size = 128
    self.patch_size = 64
    image_center = image_size // 2
    self.patch_center = self.patch_size // 2
    canvas_size = math.ceil(math.sqrt(2 * image_size ** 2))
    canvas_size += canvas_size % 2
    self.canvas_center = canvas_size // 2
    self.img_64_bound = self.canvas_center - self.patch_center


  def augmentation(self, img):
    canvas = img
    rotated = self._rotate(canvas, self.canvas_center, self.canvas_center,
                     random.random() * self.rotate_range)
    crop_x = self.img_64_bound + random.randint(-self.shift_range, self.shift_range)
    crop_y = self.img_64_bound + random.randint(-self.shift_range, self.shift_range)
    cropped = rotated[crop_y:crop_y + self.patch_size, crop_x:crop_x + self.patch_size]
    if self.flip_ud and random.randrange(2):
        cropped = np.flipud(cropped)
    if self.flip_lr and random.randrange(2):
        cropped = np.fliplr(cropped)
    # cropped = cropped.astype(
    #     np.float32) * (math.exp(random.uniform(-self.scale_log, self.scale_log)) / 255)
    # cropped -= cropped.mean()
    return cropped


  def _rotate(self, img, cx, cy, angle):
      return cv.warpAffine(img, cv.getRotationMatrix2D((cx, cy), angle, 1.0), (img.shape[1], img.shape[0]),
                          flags=cv.INTER_LINEAR)
  def to_64(self, img):
      return img[self.img_64_bound:self.img_64_bound + self.patch_size, self.img_64_bound:self.img_64_bound + self.patch_size]

def _preprocess_128_images(x_train_128):
    image_size = 128
    canvas_size = math.ceil(math.sqrt(2 * image_size ** 2))
    canvas_size += canvas_size % 2
    canvas_center = canvas_size // 2
    canvas_margin = (canvas_size - image_size) // 2

    canvases_128 = np.ndarray(shape=(x_train_128.shape[0], canvas_size, canvas_size,1), dtype=np.float32)
    canvases_128[...] = 0
    canvases_128[:, canvas_margin:canvas_margin+image_size,
                canvas_margin:canvas_margin+image_size] = x_train_128

    return canvases_128
def normalize(dataset):
    dataset_mean = np.mean(dataset.reshape((len(dataset), 64 * 64)), axis=-1).reshape(
        (len(dataset), 1, 1, 1))
    return (dataset - dataset_mean) / 255
def fix_data(flag,x_train,y_train,x_val,y_val,x_test,y_test):
    if flag:
        preprocess_config = {
            'shift_range': 5,
            'rotate_range': 360.0,
            'flip_ud': True,
            'flip_lr': True,
            'scale_range': 1.25,
        }
        data_aug = DataAug(**preprocess_config)
        x_test_cp = []
        x_val_cp = []
        for i in range(len(x_val)):
            x_val_cp.append(data_aug.to_64(x_val[i]))
        for i in range(len(x_test)):
            x_test_cp.append(data_aug.to_64(x_test[i]))
        x_test = x_test_cp
        x_val = x_val_cp
        x_test = np.array(x_test)
        x_val = np.array(x_val)
        #####################################
    data = {"x_train": x_train, "y_train": y_train, "x_val": x_val, "y_val": y_val, "x_test": x_test,
            "y_test": y_test}
    return data
def load_data(label='h', phase='aug_evaluation'):

    labels_dic = {
        'h': 'head',
        'a': 'acrosome',
        'v': 'vacuole',
    }

    assert label in labels_dic.keys(), 'Label is not correct.'

    label = labels_dic[label]

    dataset_addr = 'mhsma-dataset/mhsma'
    x_train_128_file = os.path.join(dataset_addr, 'x_{}_train.npy'.format(str(128)))
    x_valid_128_file = os.path.join(dataset_addr, 'x_{}_valid.npy'.format(str(128)))
    x_test_128_file = os.path.join(dataset_addr, 'x_{}_test.npy'.format(str(128)))
    x_valid_file = os.path.join(dataset_addr, 'x_{}_valid.npy'.format(str(64)))
    x_train_file = os.path.join(dataset_addr, 'x_{}_train.npy'.format(str(64)))
    x_test_file  = os.path.join(dataset_addr, 'x_{}_test.npy'.format(str(64)) )

    y_valid_file = os.path.join(dataset_addr,
                     'y_{}_valid.npy'.format(label))
    y_train_file = os.path.join(dataset_addr,
                     'y_{}_train.npy'.format(label))
    y_test_file  = os.path.join(dataset_addr,
                     'y_{}_test.npy'.format(label))

    x_train_128 = np.load(x_train_128_file).astype(np.float32)
    x_valid_128 = np.load(x_valid_128_file).astype(np.float32)
    x_test_128 = np.load(x_test_128_file).astype(np.float32)
    x_valid = np.load(x_valid_file).astype(np.float32)
    x_train = np.load(x_train_file).astype(np.float32)
    x_test = np.load(x_test_file).astype(np.float32)

    x_train = np.reshape(x_train, (len(x_train), 64, 64, 1))
    x_valid = np.reshape(x_valid, (len(x_valid), 64,64, 1))
    x_test = np.reshape(x_test, (len(x_test), 64,64, 1))
    x_train_128 = np.reshape(x_train_128, (len(x_train_128),128,128, 1))
    x_valid_128 = np.reshape(x_valid_128, (len(x_valid_128),128,128, 1))
    x_test_128 = np.reshape(x_test_128, (len(x_test_128),128,128, 1))

    y_valid = np.load(y_valid_file)[..., None]
    y_train = np.load(y_train_file)[..., None]
    y_test = np.load(y_test_file)[..., None]

   #x_train_128 = _preprocess_128_images(x_train_128)
   #x_valid_128 = _preprocess_128_images(x_valid_128)
   #x_test_128 = _preprocess_128_images(x_test_128)
    if phase == 'search':

        return {
                "x_train": x_train_128,
                "y_train": y_train,
                "x_train_128": x_train_128,
                'x_val_128': x_valid_128,
                "x_val": x_valid_128,
                "y_val": y_valid,
                "x_test": x_test_128,
                "y_test": y_test
                }

    elif phase == 'evaluation':
        tmp_x=np.append(x_train,x_valid,axis=0)
        tmp_x=np.append(tmp_x,x_test,axis=0)
        tmp_y=np.append(y_train,y_valid,axis=0)
        tmp_y=np.append(tmp_y,y_test,axis=0)
        return {"x": tmp_x,"y": tmp_y}
    elif phase == 'aug_evaluation':
        tmp_x=np.append(x_train_128,x_valid_128,axis=0)
        tmp_x=np.append(tmp_x,x_test_128,axis=0)
        tmp_y=np.append(y_train,y_valid,axis=0)
        tmp_y=np.append(tmp_y,y_test,axis=0)
        return {"x": tmp_x,"y": tmp_y}
    else:
        raise Exception("Value of 'phase' is not recognizable")
