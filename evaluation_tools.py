import sys
import pickle
import pandas as pd
import numpy as np
import os
import tensorflow.compat.v1 as tf
import math
import sklearn
from sklearn.metrics import roc_auc_score
tf.disable_v2_behavior()
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix
from utils.tools import *
from keras.models import load_model
from tensorflow.keras.models import load_model
from keras.optimizers import Adam
from keras import backend as K

def evaluation(preds, reals, p,verbos=True):
    c_matrix = confusion_matrix(reals, preds)
    precision = c_matrix[0, 0] / sum(c_matrix[0])
    recall = c_matrix[0, 0] / sum(c_matrix[:, 0])
    G_mean = math.sqrt(precision * recall)
    accuracy = np.sum(c_matrix.diagonal()) / np.sum(c_matrix)
    f = 1.25 * precision * recall / (.25 * precision + recall)
    TP = c_matrix[0][0]
    TN = c_matrix[1][1]
    FP = c_matrix[0][1]
    FN = c_matrix[1][0]
    MCC = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    AUC = roc_auc_score(reals, p)

    if verbos:
        print('confusion_matrix', c_matrix)
        print('precision ', precision)
        print('recall ', recall)
        print('f_0.5 ', f)
        print('G_mean', G_mean)

        print('MCC', MCC)
        print('AUC', AUC)
        print('accuracy ', accuracy)
    else:
        return accuracy+f


def model_evaluation(addr, label_index, label,type, threshold=.5, optimizer='adam',grid_search=False):
    model = load_model(addr)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mse'])
    dataset_name = 'gray'
    models_dir_path = 'models'
    model_checkpoints_dir_path = 'models/checkpoints'
    print(model.summary())
    conv_num = 0
    pool_num = 0
    for l in model.layers:
        conf = l.get_config()
        #     print(conf)
        if 'conv' in conf['name']:
            conv_num += 1
            print(conf['name'], 'filters:' + str(conf['filters']), 'strides:' + str(conf['strides']),
                  'kernel_size:' + str(conf['kernel_size']))
        elif 'max' in conf['name'] or 'average' in conf['name']:
            pool_num += 1
            print(conf['name'], 'strides:' + str(conf['strides']), 'pool_size:' + str(conf['pool_size']))
        else:
            print(conf)
    print('pool_num: ', pool_num)
    print('conv_num: ', conv_num)
    label_abbrv = {
        'h': 'head',
        'v': 'vacuole',
        'a': 'acrosome'
    }
    y_suff = label_abbrv[label]

    x_test_file_path = 'mhsma-dataset/mhsma/x_64_test.npy'
    y_test_file_path = 'mhsma-dataset/mhsma/y_' + y_suff + '_test.npy'
    x_valid_file_path = 'mhsma-dataset/mhsma/x_64_valid.npy'
    y_valid_file_path = 'mhsma-dataset/mhsma/y_' + y_suff + '_valid.npy'

    x_test = np.load(x_test_file_path)
    x_valid = np.load(x_valid_file_path)

    x_test = x_test.reshape(*x_test.shape, 1)
    y_test = np.load(y_test_file_path).astype(np.float32)

    x_valid = x_valid.reshape(*x_valid.shape, 1)
    y_valid = np.load(y_valid_file_path).astype(np.float32)

    val_preds = model.predict(x_valid, batch_size=32, verbose=0)
    test_preds = model.predict(x_test, batch_size=32, verbose=0)
    if grid_search:
        thresholds = np.arange(.2, .8, .001)
        best_acc = float('-inf')
        if type=="dmtl":
            for th in thresholds:
                acc = evaluation(val_preds[label_index] > th, y_valid,val_preds[label_index], verbos=False)
                if acc > best_acc:
                    print(th, acc)
                    best_acc = acc
                    threshold = th

        evaluation(val_preds[label_index] > threshold, y_valid,val_preds[label_index])
        evaluation(test_preds[label_index] > threshold, y_test,test_preds[label_index])
    else:
        for th in thresholds:
            acc = evaluation(val_preds > th, y_valid, val_preds, verbos=False)
            if acc > best_acc:
                print(th, acc)
                best_acc = acc
                threshold = th

        evaluation(val_preds > threshold, y_valid, val_preds)
        evaluation(test_preds > threshold, y_test, test_preds)


def evaluate_model(addr, x_test, y_test, x_valid, y_valid):
    model = load_model(addr)
    test_ev = model.evaluate(x_test, y_test, batch_size=32, verbose=0)
    val_ev = model.evaluate(x_valid, y_valid, batch_size=32, verbose=0)
    return test_ev[0], test_ev[1], val_ev[0], val_ev[1]