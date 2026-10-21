import torch
import torchvision.datasets as dataset
import numpy as np
from PreProcessSMAP import *
from PreProcessMSL import *
from PreProSMAP_MSL import *
from preprocess_wadi import *
import pandas as pd
from scipy import io
from utils import *
from pathlib import Path
#from smd_preprocess import *
from msl_preprocess import *
from smap_preprocess import *
from preprocess_psm import *
from preprocess_swat import *
from preprocess_smd import *


def load_data(data_name):
    path = 'DATA/'
    data_path = path + data_name + '/'
    if data_name == 'smd':
        train_data, train_label, test_data, test_label  = smd_train_test_load(data_path) #smd_train_test_split(data_path)
    elif data_name == 'smap':
        train_data, train_label, test_data, test_label = smap_train_test_load(data_path) #smap_train_test_split(data_path)
    elif data_name == 'msl':
        train_data, train_label, test_data, test_label = msl_train_test_split(data_path) #msl_train_test_load(data_path) #
    elif data_name == 'wadi':
        train_data, train_label, test_data, test_label = wadi_train_test_split(data_path)
    elif data_name == 'psm':
        train_data, train_label, test_data, test_label = psm_train_test_load(data_path)
    elif data_name == 'swat':
        train_data, train_label, test_data, test_label = swat_train_test_load(data_path)
    
    #trainset = CustomDataset(train,train_label)
    #testset = CustomDataset(test,test_label)
    return train_data, train_label, test_data, test_label # Explicitly return all four values    

        
def smd_train_test_split(path):
    print('Path : ', path)
    path = path + 'machine-1-1.csv'
    data = pd.read_csv(path, header=1)
    
    # Split features and labels
    samples = data.iloc[:, :-1].values  # All columns except the last
    labels = data.iloc[:, -1].values     # Last column only

    inliers = samples[labels == 0]  # 3679 norm
    outliers = samples[labels == 1]  # 93 anom

    print('smd Inliers: ', inliers.shape)
    print('smd outliers: ', outliers.shape)

    train_data, train_label, test_data, test_label=train_test_split(inliers,outliers)
    print('smd Train Data Shape ', train_data.shape)
    print('smd Train Label Shape: ', train_label.shape)
    print('smd Test Data Shape: ', test_data.shape)
    print('smd Test Label Shape: ', test_label.shape)
    return train_data, train_label, test_data, test_label
    #return samples, labels

def smap_train_test_split(base_dir):
    #base_dir = '/content/drive/MyDrive/Udesha/DATASET/SMAP/';
    train_data,test_data,test_label,train_label = preprocess_smap_msl_trim(base_dir)

    print('SMAP Train Data : ', train_data.shape)
    print('SMAP Test Data : ', test_data.shape)
    print('SMAP Test Labels : ', test_label.shape)
    print('SMAP Train labels : ', train_label.shape)
    return train_data, train_label, test_data, test_label

def msl_train_test_split(base_dir):
    train_data,test_data,test_label,train_label = preprocess_msl_trim(base_dir)  

    print('MSL Train Data : ', train_data.shape)
    print('MSL Test Data : ', test_data.shape)
    print('MSL Test Labels : ', test_label.shape)
    print('MSL Train labels : ', train_label.shape)
    return train_data, train_label, test_data, test_label

def train_test_split(inliers,outliers):
    num_split = len(inliers) // 2
    train_data = inliers[:num_split]
    train_label = np.zeros(num_split)
    test_data = np.concatenate([inliers[num_split:],outliers],0)

    test_label = np.zeros(test_data.shape[0])
    test_label[num_split:]=1
    return train_data, train_label, test_data, test_label

def wadi_train_test_split(base_dir):
  TRAIN_CSV = Path(base_dir + "WADI_trainingdata.csv")
  TEST_CSV  = Path(base_dir + "WADI_attackdata.csv")
  
  train_data, train_label, test_data, test_label = preprocess_wadi(
        train_csv=TRAIN_CSV,
        test_csv=TEST_CSV,
    )

  print('WADI Train Data : ', train_data.shape)
  print('WADI Test Data : ', test_data.shape)
  print('WADI Test Labels : ', test_label.shape)
  print('WADI Train labels : ', train_label.shape)
  
  return train_data, train_label, test_data, test_label

def smd_train_test_load_pre(base_dir):
    TRAIN_CSV  = Path(base_dir + "train/machine-3-11.txt")
    TEST_CSV   = Path(base_dir + "test/machine-3-11.txt")
    LABELS_CSV = Path(base_dir + "test_label/machine-3-11.txt")

    train_data, train_label, test_data, test_label = load_smd(
        train_csv  = TRAIN_CSV,
        test_csv   = TEST_CSV,
        labels_csv = LABELS_CSV,
    )

    print('SMD Train Data : ', train_data.shape)
    print('SMD Test Data : ', test_data.shape)
    print('SMD Test Labels : ', test_label.shape)
    print('SMD Train labels : ', train_label.shape)
  
    return train_data, train_label, test_data, test_label

def smd_train_test_load(base_dir):
    train_dir = base_dir + "train"
    test_dir  = base_dir + "test"
    label_dir = base_dir + "test_label"

    train_data, train_label, test_data, test_label = load_smd(
        train_dir=train_dir,
        test_dir=test_dir,
        label_dir=label_dir,
        verbose=True,
    )

    print('SMD Train Data : ', train_data.shape)
    print('SMD Test Data : ', test_data.shape)
    print('SMD Test Labels : ', test_label.shape)
    print('SMD Train labels : ', train_label.shape)
  
    return train_data, train_label, test_data, test_label
  
def msl_train_test_load(base_dir):
  TRAIN_DIR   = base_dir + "train"           # folder containing C-1.npy, C-2.npy …
  TEST_DIR    = base_dir + "test"            # folder containing C-1.npy, C-2.npy …
  ANOMALY_CSV = base_dir + "labeled_anomalies.csv"
  # ───────────────────────────────────────────────────────────────────────

  train_data, train_label, test_data, test_label = load_msl(
      train_dir   = TRAIN_DIR,
      test_dir    = TEST_DIR,
      anomaly_csv = ANOMALY_CSV,
  )
  print('MSL Train Data : ', train_data.shape)
  print('MSL Test Data : ', test_data.shape)
  print('MSL Test Labels : ', test_label.shape)
  print('MSL Train labels : ', train_label.shape)

  return train_data, train_label, test_data, test_label

def smap_train_test_load(base_dir):
  TRAIN_DIR   = base_dir + "train"           # folder containing C-1.npy, C-2.npy …
  TEST_DIR    = base_dir + "test"            # folder containing C-1.npy, C-2.npy …
  ANOMALY_CSV = base_dir + "labeled_anomalies.csv"
  # ───────────────────────────────────────────────────────────────────────

  train_data, train_label, test_data, test_label = load_smap(
      train_dir   = TRAIN_DIR,
      test_dir    = TEST_DIR,
      anomaly_csv = ANOMALY_CSV,
  )
  print('SMAP Train Data : ', train_data.shape)
  print('SMAP Test Data : ', test_data.shape)
  print('SMAP Test Labels : ', test_label.shape)
  print('SMAP Train labels : ', train_label.shape)

  return train_data, train_label, test_data, test_label

def psm_train_test_load(base_dir):
  print('BAse DIR Path : ', base_dir)
  train_path = base_dir + "train.csv"
  test_path  = base_dir + "test.csv"
  label_path = base_dir + "test_label.csv"

  train_data, train_label, test_data, test_label = load_psm(
      train_path=train_path,
      test_path=test_path,
      label_path=label_path,
      verbose=True,
  )
  print('PSM Train Data : ', train_data.shape)
  print('PSM Test Data : ', test_data.shape)
  print('PSM Test Labels : ', test_label.shape)
  print('PSM Train labels : ', train_label.shape)

  return train_data, train_label, test_data, test_label

def swat_train_test_load(base_dir):
  normal_v0 = base_dir + "SWaT_Dataset_Normal_v0.xlsx"
  normal_v1 = base_dir + "SWaT_Dataset_Normal_v1.xlsx"
  attack_v0 = base_dir + "SWaT_Dataset_Attack_v0.xlsx"

  train_data, train_label, test_data, test_label = load_swat(
      normal_v0_path=normal_v0,
      normal_v1_path=normal_v1,
      attack_v0_path=attack_v0,
      verbose=True,
  )
  print('SWaT Train Data : ', train_data.shape)
  print('SWaT Test Data : ', test_data.shape)
  print('SWaT Test Labels : ', test_label.shape)
  print('SWaT Train labels : ', train_label.shape)

  return train_data, train_label, test_data, test_label
