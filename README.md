# Graph Deviation Network (GDN) — Anomaly Detection on SMAP & SMD

This repository contains our implementation and experiments for the **MVA 2025/2026 — Machine Learning for Time Series** mini-project.  
We study the **Graph Deviation Network (GDN)** method by Deng & Hooi (KDD 2021) and apply it to two multivariate time-series datasets: **NASA SMAP** and **Server Machine Dataset (SMD)**.  
GDN learns a graph of dependencies between sensors and detects anomalies as **deviations from the learned graph-based forecasting behaviour**.

---

## Datasets

We use the implementations provided in `torch_timeseries`:

- **SMAP** (NASA telemetry anomaly benchmark)  
  - 25 sensors (1 continuous, 24 binary)  
  - Expert-labelled anomalous segments  
  - Training split assumed anomaly-free

- **SMD** (Server Machine Dataset)  
  - 38 continuous sensors  
  - Single long multivariate series with labelled anomalous segments  
  - Training split assumed anomaly-free

Both datasets are automatically downloaded by the loading utilities; no manual download is required.

---

## Model

We reuse the original **GDN architecture** and adapt it to our custom data loaders and evaluation code.  
GDN learns:

- **Sensor embeddings** in a latent space
- A **Top-k similarity graph** between sensors (cosine similarity on embeddings)
- A **graph attention–based forecasting network** that predicts the next time step from a sliding window of past observations
- An **anomaly score** based on normalized prediction errors and max-pooling over sensors

Training is done with **MSE loss** on the one-step-ahead forecast, following the original paper.

---

## Experiments

Beyond reproducing the original configuration of GDN, we perform several additional experiments on both datasets:

### Common experiments (SMAP & SMD)

- **Data analysis and preprocessing**
  - Basic statistics, autocorrelation, spectra
  - Decision not to detrend or denoise, based on signal inspection

- **Linear baselines**
  - **VAR(100)**: multivariate autoregressive model on all sensors  
  - **AR(100)**: univariate autoregressive model on a single sensor

- **Reproduction of the original GDN configuration**
  - Same hyper-parameters as in the paper (`paper_config`)

- **Anomalies in the training set**
  - Train GDN on data that contain anomalies (`anom_in_train`) to study robustness to contaminated training sets

- **Effect of the context window size**
  - Vary the sliding-window length $ w $ (e.g. $ w \in \{5, 10, 20, 50, 75, 100, 150, 200\} $)  
  - Study its impact on AUC and the difference between SMAP and SMD

- **Point-wise vs sequence-wise evaluation**
  - Point-wise: metrics computed per time step  
  - Sequence-wise: merge each anomalous segment into a single point (keep the maximum score on the segment) before computing metrics

### SMAP-specific experiments

- **Mixed loss / score for continuous and binary sensors (`mix_cont/bin`)**
  - Continuous channel: MSE loss and absolute error
  - Binary channels: BCE loss on logits and sigmoid-transformed errors
  - Adapted anomaly scoring to account for heterogeneous sensor types

### Overall findings

- On **SMAP**, all GDN variants perform poorly in strict point-wise AUC and remain below the linear baselines; the dataset’s mostly binary nature makes forecasting-based detection difficult.
- On **SMD**, all models achieve higher AUC, and **VAR(100)** slightly outperforms GDN and AR, suggesting that simple linear models with full multivariate context can be very competitive.
- Injecting anomalies into the training data strongly degrades performance, confirming the importance of a clean training set.
- The context window has opposite effects on SMAP and SMD: small windows are best on SMAP, whereas larger windows improve performance on SMD.

---

## Code reuse

In this project, we **reuse the original implementation of the GDN model architecture** and **adapt the training and testing functions**.  
All other components are **implemented by us**.

---

## How to Run

Open the main notebook of this repository (for example:

```bash
jupyter notebook mini_project_gdn.ipynb
```

---

### References
	•	Ailin Deng & Bryan Hooi — Graph Neural Network-Based Anomaly Detection in Multivariate Time Series (GDN), KDD 2021
	•	Franco Scarselli et al. — The Graph Neural Network Model, IEEE TNN, 2009
	•	NASA SMAP Anomaly Benchmark
	•	Server Machine Dataset (SMD) benchmark

