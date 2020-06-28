
import argparse
import os
import sys
print('sys.argv: ', sys.argv)
sys.stdout.flush()
git clone https://github.com/soroushj/mhsma-dataset.git
from modules.Sampler import Sampler
import tensorflow as tf
import os
from utils.evaluation_tools import *
import tensorflow as tf
from modules.MyModel import MyModel
from modules.LoadData import load_data
from modules.DMTL import DMTL
from modules.DTL import DTL
import os
import h5py
from tensorflow.keras.models import save_model,load_model


def get_args():
  
  parser = argparse.ArgumentParser(description='Ensemble Transfer Learning Algorithm')

  parser.add_argument('-t', dest='type', default='dmtl',
                      type=str, nargs='?',
                      help='Type of Netwrok')
  parser.add_argument('-e', dest='epochs', default=300,
                    type=int, nargs='?',
                    help='Number of epochs each CNN should train on')
  parser.add_argument('--phase', dest='phase', default=1,
                    type=int, nargs='?',
                    help='DMTL phase')
  parser.add_argument('--label', dest='label', default='h',
                      type=str, nargs='?',
                      help="Chosen Label: ['h', 'v', 'a'] ")
  parser.add_argument('--activation', dest='activation', default='sigmoid',
                      type=str, nargs='?',
                      help="Chosen Label: ['h', 'v', 'a'] ")
  parser.add_argument('--model', dest='model', default='vgg_19',
                      type=str, nargs='?',
                      help="Chosen model:[densenet,resnet,resnetV2,inceptionV3]"
                           "\n densenet=[121,169,201]"
                           "\n densenet=[19]"
                           "\n resnet=[50,101,152]"
                           "\n resnetV2=[50,101,152]"
                           "inceptionV3=nothing"
                           "\n usage:[name]_[version]"
                           "\n example of use:!python --model densenet_121")
  return parser.parse_args()
def get_model(arg):
    name,version=arg.split("_")
    list={}
    list["vgg"]={"19":tf.keras.applications.vgg19.VGG19}
    list["densenet"]={"121":tf.keras.applications.densenet.DenseNet121,"169":tf.keras.applications.densenet.DenseNet169,"201":tf.keras.applications.densenet.DenseNet201}
    list["resnet"]={"50":tf.keras.applications.resnet.ResNet50,"101":tf.keras.applications.resnet.ResNet101,"152":tf.keras.applications.resnet.ResNet152}
    list["resnetV2"]={"50":tf.keras.applications.resnet_v2.ResNet50V2,"101":tf.keras.applications.resnet_v2.ResNet101V2,"152":tf.keras.applications.resnet_v2.ResNet152V2}
    list["inceptionV3"]={"":tf.keras.applications.inception_v3.InceptionV3}
    name=name.lower()
    return list[name][version]

def _main(interval,log_path,epochs,label,model,type,phase):

    checkpoint_file_name = str(interval[0])+"_"+str(interval[1])

    if os.path.isfile(checkpoint_file_name + ".txt"): # Loading the checkpoint
      with open(checkpoint_file_name + ".txt","r") as f:
        interval[0] += int(f.read())
    activation = "sigmoid"
    regularizition = 0.0
    dropout = 0.0
    number_of_neurons = 1024
    loss = "binary_crossentropy"
    label="a"
    if type=="dtl":
        params = {"agumentation": False, "scale": False, "dense_activation": activation,
                  "regularizition": regularizition, "dropout": dropout, "optimizer": "adadelta",
                  "number_of_dense": 1, "balancer": "None", "loss": loss, "batch_size": 64, "nn": number_of_neurons}

        model = DTL(params=params, base_model=model,label=label)
        model.train(epochs, load_best_weigth=True, verbose=1, TensorB=True, name_of_best_weight="t.h5", phase="train")
        ans = model.evaluate()
        print(ans)
    else:
        activation = ["selu", "selu", "selu"]  
        regularizition = [0.1, 0.0, 0.1]
        dropout = [0.0, 0.9, 0.0]
        number_of_neurons = [1024, 1024, 1024]
        loss = "binary_crossentropy"
        label_index = 1
        epochs = 500
        phase = 2
        params = {"agumentation": False, "scale": False, "dense_activation": activation,
                  "regularizition": regularizition,
                  "dropout": dropout, "optimizer": "adadelta", "number_of_dense": [1, 1, 1], "balancer": "None",
                  "batch_size": 64, "nn": number_of_neurons}
        model = DMTL(params=param,base_model=model,label_index=label_index,loss='binary_crossentropy',second_model=mo) if phase==2 else DMTL(params=params, base_model=model)
        model.train(epochs, load_best_weigth=False, verbose=1, TensorB=True,name_of_best_weight="t.h5",phase="train")
        model.evaulate()


def main():
  _main([args.low_interval, args.high_interval], args.log_path, args.epochs,args.label,args.model,args.type,args.phase)
  sys.stdout.flush()

if __name__ == '__main__':
  main()
