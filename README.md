# CSGL-AD : Cross-Scale Graph Learning for Anomaly Detection on Multivariate Time Series
Anomaly Detection (AD) in multivariate time series (MTS) is a critical task for industrial IoT systems, water infrastructure, and server monitoring, where undetected faults can lead to severe system failures. Existing deep learning methods have three key limitations: modelling temporal dependencies at a single fixed resolution, constructing spatial graphs from randomly initialized embeddings without multi-scale awareness, and applying graph-based rea-soning exclusively to inter-sensor dependencies while neglecting structural relationships between temporal observations across scales. To address these limitations, we propose Cross-Scale Graph Learning for Anomaly Detection (CSGL-AD), a novel spatio-temporal graph neural network framework that builds semantically meaningful graph representations at multiple temporal scales before applying graph-based reasoning. CSGL-AD introduces a hier-archical multi-scale temporal encoder that decomposes each input window into multiple resolutions and provides learned resolution-specific embed-dings to initialize both the temporal and spatial GNN streams. A temporal GNN connects time-window segments via structural overlap across scales. In contrast, a spatial GNN connects each sensor's scale embeddings via same-channel edges, propagating a sensor's own representation across resolutions; an inter-sensor graph attention module then models dependencies between different sensors over a sparsified, top-k sensor-dependency graph within each scale. A multi-head anomaly scorer computes three complementary re-construction-error signals, each weighted by its scale importance. Extensive experiments across five widely used benchmark datasets show that CSGL-AD consistently achieves strong AUROC, surpassing strong baselines by up to 28 percentage points, and generalises robustly across diverse anomaly types and application domains.

<img width="910" height="508" alt="image" src="https://github.com/user-attachments/assets/82b86aed-701e-47c4-a48b-ba0adddea587" />

# Quickstart

Given a python environment (note: this project is fully tested under python 3.13), install the fllowing dependency:

```
!pip install torch-geometric
```

Please run the following command to train and test the model:

```
!python run.py --dataset-name $1
```

Note: replace $# with available options

## How to Use
1. When using your own data, please put your data files under [DATA](CSGL-AD/DATA).

2. Add your data loader to the function ''load_data'' in the [LoadData.py](CSGL-AD/LoadData.py).
