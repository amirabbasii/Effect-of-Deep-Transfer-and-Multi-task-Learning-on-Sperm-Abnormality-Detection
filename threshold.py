import sys
import numpy as np
import math
import tensorflow.compat.v1 as tf
import argparse
from modules.LoadData import load_data
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix
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


def model_evaluation(addr, label,type, threshold=.5, optimizer='adam',grid_search=False):
    model = load_model(addr)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mse'])
    data = load_data(label=label, phase="search")
    val_preds = model.predict(data['x_val'], batch_size=32, verbose=0)
    test_preds = model.predict(data['x_test'], batch_size=32, verbose=0)
    if grid_search:
        thresholds = np.arange(.2, .8, .001)
        best_acc = float('-inf')
        if type=="dmtl":

            label_index = ['a', 'v', 'h'].index(label)
            for th in thresholds:
                acc = evaluation(val_preds[label_index] > th, data["y_val"],val_preds[label_index], verbos=False)
                if acc > best_acc:
                    best_acc = acc
                    threshold = th

            evaluation(val_preds[label_index] > threshold, data["y_val"],val_preds[label_index])
            evaluation(test_preds[label_index] > threshold, data["y_test"],test_preds[label_index])
        else:
            for th in thresholds:
                acc = evaluation(val_preds > th, y_valid, val_preds, verbos=False)
                if acc > best_acc:
                    print(th, acc)
                    best_acc = acc
                    threshold = th

            evaluation(val_preds > threshold, data["y_val"], val_preds)
            evaluation(test_preds > threshold, data["y_test"], test_preds)
        print("best threshold:",threshold)


def get_args():
    parser = argparse.ArgumentParser(description='Parser')

    parser.add_argument('-addr', dest='address',
                        type=str, nargs='?',
                        help='model address')
    parser.add_argument('-t', dest='type', default='dmtl',
                        type=str, nargs='?',
                        help='Type of Netwrok')
    parser.add_argument('-l', dest='label', default='a',
                        type=str, nargs='?',
                        help='label')




    return parser.parse_args()
def main():
  args=get_args()
  model_evaluation(args.address,args.label,args.type,grid_search=True)

if __name__ == '__main__':
  main()
