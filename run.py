import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import os
#from Spatial_Temporal_GNN_MoE import ScaleGNNConfig
#from Spatial_Temporal_GNN_MoE import run_pipeline
# final correct from Temp_Spatial_GNN4 import ScaleGNNConfig
# final correct from Temp_Spatial_GNN4 import run_pipeline
from Temp_Spatial_Inter_New import ScaleGNNConfig
from Temp_Spatial_Inter_New import run_pipeline
import torch.nn.functional as F
from smd_preprocess import *

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define path (already changed in notebook, but keeping for standalone script execution)
path_in_script = "/content/drive/MyDrive/Udesha/ScaledSTGNN_V2/"
os.chdir(path_in_script)

# Import local modules
from DataLoader import load_data
from MultiScaleModel import Net3 # Assuming Net3 is the correct class name
from TemporalGCN import CrossScaleTemporalGCNModel

# Argument parser (simplified for Colab context if needed, or keep as is)
parser = argparse.ArgumentParser(description='ScaledSTGNN Anomaly Detection')
parser.add_argument('--dataset-name', type=str, default='smd', help='Dataset name (e.g., smd, smap, msl)')
parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
parser.add_argument('--hidden-dim', type=int, default=64, help='Hidden dimension for GCN layers')
parser.add_argument('--kernels', type=int, default=64, help='Number of kernels for CNN layers in Net3')
parser.add_argument('--fine-len', type=int, default=3, help='Fine scale length')
parser.add_argument('--mid-len', type=int, default=6, help='Mid scale length')
parser.add_argument('--coarse-len', type=int, default=12, help='Coarse scale length')
parser.add_argument('--topk', type=int, default=5, help='Top K for graph construction (if applicable)')

# In a Colab execution, we might not pass args via command line, so parse a dummy if not present
try:
    args = parser.parse_args()
except SystemExit:
    args = parser.parse_args([]) # For Colab, parse no arguments initially

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
dataset_name = args.dataset_name
# Load data
#train_data, train_label, test_data, test_label = load_data(args.dataset_name)
if(dataset_name != 'smd'):
  train_data, train_label, test_data, test_label = load_data(args.dataset_name)

  print(f"{args.dataset_name} Train Data Shape {train_data.shape}")
  print(f"{args.dataset_name} Train Label Shape: {train_label.shape}")
#print(f"{args.dataset_name} Test Data Shape: {test_data.shape}")
#print(f"{args.dataset_name} Test Label Shape: {test_label.shape}")


# Convert numpy arrays to PyTorch tensors
# The input to Net3 needs to be (Batch, Time, Features)

if(dataset_name == 'wadi'):
  split_idx    = int(len(test_data) * 0.7)
  val_data     = test_data[:split_idx]
  val_labels   = test_label[:split_idx]
  test_data_pt  = test_data[split_idx:]
  test_label_pt = test_label[split_idx:]
  train_data_pt = train_data
  train_label_pt = train_label
  batch_size = 320
  learning_rate = 35e-4
elif (dataset_name == 'smd'):
  #split_idx1 = int(len(test_data) * 0.2)
  #train_extra_data = test_data[:split_idx1]
  #test_label_new = test_label[split_idx1:]
  #test_dataset = test_data[split_idx1:]
  #split_idx    = int(len(test_dataset) * 0.5)   # first 30% of test → val 
  #val_data     = test_dataset[:split_idx]
  #val_labels   = test_label_new[:split_idx]
  #test_data_pt  = test_dataset[split_idx:]
  #test_label_pt = test_label_new[split_idx:]
  #train_data_pt = np.concatenate([train_data, train_extra_data], axis=0)
  #train_label_pt = np.concatenate([train_label, np.zeros(split_idx1 + 1, dtype=int)])

  #split_idx    = int(len(test_data) * 0.3)
  #val_data     = test_data[:split_idx]
  #val_labels   = test_label[:split_idx]
  #test_data_pt  = test_data[split_idx:]
  #test_label_pt = test_label[split_idx:]
  #train_data_pt = train_data
  #train_label_pt = train_label
  batch_size = 256
  learning_rate = 2e-3
elif (dataset_name == 'msl'):
  #split_idx1 = int(len(test_data) * 0.5)
  #train_extra_data = test_data[split_idx1:]
  #test_label_new = test_label[:split_idx1]
  #test_dataset = test_data[:split_idx1]
  split_idx    = int(len(test_data) * 0.05)   # first 30% of test → val 
  val_data     = test_data[:split_idx]
  val_labels   = test_label[:split_idx]
  test_data_pt  = test_data[split_idx:]
  test_label_pt = test_label[split_idx:]
  train_data_pt = train_data
  train_label_pt = train_label
  #train_data_pt = np.concatenate([train_data, train_extra_data], axis=0)
  #train_label_pt = np.concatenate([train_label, np.zeros(split_idx1 + 1, dtype=int)])
  batch_size = 256
  learning_rate = 2e-3
elif (dataset_name == 'smap'):
  print('In split Else')
  split_idx    = int(len(test_data) * 0.6)
  train_data_pt = train_data #torch.from_numpy(train_data).float().to(device)
  test_data_pt = test_data #torch.from_numpy(test_data).float().to(device)
  train_label_pt = train_label #torch.from_numpy(train_label).float().to(device)
  test_label_pt = test_label #torch.from_numpy(test_label).float().to(device)
  val_data     = test_data_pt[split_idx:] #test_data_pt[:split_idx]
  val_labels   = test_label_pt[split_idx:] #test_label_pt[:split_idx]
  test_data_pt  = test_data_pt[:split_idx] #test_data_pt[split_idx:]
  test_label_pt = test_label_pt[:split_idx]    
  #val_data = test_data_pt #when without splitted
  #val_labels = test_label_pt #when without splitted
  batch_size = 256
  learning_rate = 2e-3
elif (dataset_name == 'psm'):
  split_idx    = int(len(test_data) * 0.7)   # first 30% of test → val 
  val_data     = test_data[:split_idx]
  val_labels   = test_label[:split_idx]
  test_data_pt  = test_data[split_idx:]
  test_label_pt = test_label[split_idx:]
  train_data_pt = train_data
  train_label_pt = train_label
  batch_size = 256
  learning_rate = 2e-3
elif (dataset_name == 'swat'):
  split_idx    = int(len(test_data) * 0.3)   # first 30% of test → val 
  val_data     = test_data[:split_idx]
  val_labels   = test_label[:split_idx]
  test_data_pt  = test_data[split_idx:]
  test_label_pt = test_label[split_idx:]
  train_data_pt = train_data
  train_label_pt = train_label
  batch_size = 640
  learning_rate = 5e-3

if(dataset_name != 'smd'):
  train_shape = test_data_pt.shape
  num_of_features = train_shape[1]


  print("Shape      :", train_label_pt.shape)
  print("dtype      :", train_label_pt.dtype)
  print("Unique vals:", np.unique(train_label_pt))
  print("Sum        :", train_label_pt.sum())
  print("Sample     :", train_label_pt[:20])
  print("Num of features     :", num_of_features)

#for wadi best - fine_span=10, mid_span=20, coarse_span=60
cfg = ScaleGNNConfig(
        window_size=12, stride=2,
        fine_span=3, mid_span=6, coarse_span=12,
        #fine_span=10, mid_span=20, coarse_span=60,
        hidden_dim=128, gat_heads=4, num_gnn_layers=2,
        reconstruction_hidden=64, fusion_dim=64,
        use_inter_sensor_attn=True,      # inter-sensor graph attention
        inter_sensor_mode="hybrid",      #  # inter-sensor "learned" | "correlation" | "hybrid"
        inter_sensor_top_k=5,  # inter-sensor
        mean_score_weight=0.3,   # CHANGE-A
        std_score_weight=0.2,
        corr_score_weight=0.5,
        batch_size=batch_size,
        num_epochs=80,
        learning_rate=learning_rate,
        min_precision=0.20,      # CHANGE-C
        fbeta=0.5,
        threshold_percentile=99.0,
    )

if(dataset_name == 'smd'):
  machine_names = [f"machine-{g}-{i}" for g, count in [(1,8),(2,9),(3,11)] for i in range(1, count+1)]
  all_metrics = []
  all_auroc = []
  all_aff1 = []
  all_ttime = []
  all_ttimeepc = []
  all_mem = []
  all_inftime = []
  all_infwin = []
  all_meminf = []

  for machine in machine_names:
      path = 'DATA/'
      data_path = path + dataset_name + '/'
      TRAIN_CSV  = Path(data_path + "train/" + machine + ".txt")
      TEST_CSV   = Path(data_path + "test/" + machine + ".txt")
      LABELS_CSV = Path(data_path + "test_label/" + machine + ".txt")

      result = load_smd(
          train_csv  = TRAIN_CSV,
          test_csv   = TEST_CSV,
          labels_csv = LABELS_CSV,
      )
      if result is None:
        print(f"  [SKIP] Skipping machine — one or more files missing.")
        continue   # if inside a loop, or use 'pass' / skip logic as appropriate
      train_data, train_label, test_data, test_label = result
      print(f"{machine} Train Data Shape {train_data.shape}")
      print(f"{machine} Train Label Shape: {train_label.shape}")
      split_idx    = int(len(test_data) * 0.3)
      val_data     = test_data[:split_idx]
      val_labels   = test_label[:split_idx]
      test_data_pt  = test_data[split_idx:]
      test_label_pt = test_label[split_idx:]
      train_data_pt = train_data
      train_label_pt = train_label

      train_shape = test_data_pt.shape
      num_of_features = train_shape[1]


      print("Shape      :", train_label_pt.shape)
      print("dtype      :", train_label_pt.dtype)
      print("Unique vals:", np.unique(train_label_pt))
      print("Sum        :", train_label_pt.sum())
      print("Sample     :", train_label_pt[:20])
      print("Num of features     :", num_of_features)

      model, threshold, metrics = run_pipeline(
          train_data_pt,   train_label_pt,
          val_data,     val_labels,
          test_data_pt,    test_label_pt,
          cfg=cfg, dataset_name=f"SMD_{machine}",
      )
      all_metrics.append(metrics['pa_f1'])
      all_auroc.append(metrics["auroc"])
      all_aff1.append(metrics["aff_f1"])
      all_ttime.append(metrics["train_time_s"])
      all_ttimeepc.append(metrics["train_time_s_per_epoch"])
      all_mem.append(metrics["peak_mem_train_gb"])
      all_inftime.append(metrics["inference_time_s"])
      all_infwin.append(metrics["inference_ms_per_window"])
      all_meminf.append(metrics["peak_mem_inference_gb"])
      
  print(f"Macro-avg F1 : {np.mean(all_metrics):.4f} ± {np.std(all_metrics):.4f}")
  print(f"Macro-avg AUROC : {np.mean(all_auroc):.4f} ± {np.std(all_auroc):.4f}")
  print(f"Macro-avg aff-f1 : {np.mean(all_aff1):.4f} ± {np.std(all_aff1):.4f}")
  print(f"Macro-avg train time : {np.mean(all_ttime):.2f}s" f"({np.mean(all_ttimeepc):.3f}s/epoch, "
          f"peak_mem={np.mean(all_mem):.3f}GB)")
  print(f"Macro-avg Inference time    : {np.mean(all_inftime):.4f}s "
          f"({np.mean(all_infwin):.4f}ms/window, "
          f"peak_mem={np.mean(all_meminf):.3f}GB)")


else:
  model, threshold, metrics = run_pipeline(
      train_data_pt,   train_label_pt,
      val_data,     val_labels,
      test_data_pt,    test_label_pt,
      cfg=cfg,
      dataset_name=dataset_name,   # → RESULT/MSL_<timestamp>.log + .json
  )
  print(f"\nFinal test PA-F1  : {metrics['pa_f1']:.4f}")
  print(f"Final test Aff-F1 : {metrics['aff_f1']:.4f}")
  print(f"Final test AUROC  : {metrics['auroc']:.4f}")
  print(f"Train time        : {metrics['train_time_s']:.2f}s "
        f"({metrics['train_time_s_per_epoch']:.3f}s/epoch, "
        f"peak_mem={metrics['peak_mem_train_gb']:.3f}GB)")
  print(f"Inference time    : {metrics['inference_time_s']:.4f}s "
        f"({metrics['inference_ms_per_window']:.4f}ms/window, "
        f"peak_mem={metrics['peak_mem_inference_gb']:.3f}GB)")