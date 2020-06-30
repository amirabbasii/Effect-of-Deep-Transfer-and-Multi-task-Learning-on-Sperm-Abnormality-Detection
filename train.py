
import argparse
import os
import sys
from tensorflow.keras.models import save_model, load_model
import ast
print('sys.argv: ', sys.argv)
sys.stdout.flush()

from modules.Sampler import Sampler
import tensorflow as tf
import os
# from utils.evaluation_tools import *
import tensorflow as tf
from modules.MyModel import MyModel
from modules.LoadData import load_data
from modules.DMTL import DMTL
from modules.DTL import DTL
import os
import h5py
from tensorflow.keras.models import save_model,load_model


def get_args():
  
  parser = argparse.ArgumentParser(description='Parser')

  parser.add_argument('-k_fold', dest='k_fold', default="False",
                      type=str, nargs='?',
                      help='K fold')
  parser.add_argument('-t', dest='type', default='dmtl',
                      type=str, nargs='?',
                      help='Type of Netwrok')

  parser.add_argument('--second_model', dest='second_model_address', default='dmtl',
                      type=str, nargs='?',
                      help='Type of Netwrok')
  parser.add_argument('-e', dest='epochs', default=300,
                    type=int, nargs='?',
                    help='Number of epochs each CNN should train on')
  parser.add_argument('--phase', dest='phase', default=1,
                    type=int, nargs='?',
                    help='DMTL phase')

  parser.add_argument('-w', dest='bwn', default='w.h5',
                      type=str, nargs='?',
                      help="name of best weight")                 
  parser.add_argument('-label', dest='label', default='h',
                      type=str, nargs='?',
                      help="Chosen Label: ['h', 'v', 'a'] ")
  parser.add_argument('-model', dest='model', default='vgg_19',
                      type=str, nargs='?',
                      help="Chosen model:[densenet,vgg]"
                           "\n resnet=[50,101,152]"
                           "\n vgg=[16,19]"
                           "\n usage:[name]_[version]")
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
    if name in list.keys():
        return list[name][version]
    else:
        return load_model(arg)

def _main(epochs,label,model_name,type,phase,name_of_best_weight,second_model_address,k_fold):
    model=get_model(model_name)

    if type=="dtl":
        with open("dtl.txt", 'r') as file:
            s = file.readline()
            params=ast.literal_eval(s)
        if k_fold=="True":
          ans=DTL.k_fold(5,label, epochs, params, load_best_weigth=True, verbose=1, TensorB=True, name_of_best_weight=name_of_best_weight,base_model=model)
          print(ans)
        else:
          model = DTL(params=params, base_model=model,label=label)
          model.train(epochs, load_best_weigth=True, verbose=1, TensorB=True, name_of_best_weight=name_of_best_weight, phase="train")
          ans = model.evaluate()
          print(ans)
    else:
        with open("dmtl.txt", 'r') as file:
            s1, s2 = file.readlines()
            params1 = ast.literal_eval(s1)
            params2 = ast.literal_eval(s2)
        model = get_model(model_name)
        if k_fold=="True":
            ans=DMTL.k_fold(5, epochs, params1,params2, load_best_weigth=True, verbose=1, TensorB=True, name_of_best_weight=name_of_best_weight,base_model=model)
            print(ans)
        else:
          if phase==1:
            model = DMTL(params=params1,base_model=model,label=label,loss='binary_crossentropy',second_model=None,phase=1)
          else:
            mo=load_model(second_model_address)
            model = DMTL(params=params2,base_model=model,label=label,loss='binary_crossentropy',second_model=mo,phase=2)
        
          model.train(epochs, load_best_weigth=True, verbose=1, TensorB=True,name_of_best_weight=name_of_best_weight,phase="train",save_m=True)
          ans=model.evaluate()
          print(ans)


def main():
  args=get_args()
  _main(args.epochs,args.label,args.model,args.type,args.phase,args.bwn,args.second_model_address,args.k_fold)
  sys.stdout.flush()

if __name__ == '__main__':
  main()
