# Graph Deviation Network (GDN) — Anomaly Detection on SMAP & SMD

This repository contains our implementation and experiments for the  
**MVA 2025/2026 — Machine Learning for Time Series** mini-project.

We study the **Graph Deviation Network (GDN)** method proposed by Deng & Hooi (KDD 2021) for anomaly detection in multivariate time series, and apply it to **two datasets not considered in the original paper**: **NASA SMAP** and **Server Machine Dataset (SMD)**.

GDN learns a graph of dependencies between sensors and detects anomalies as  
**deviations from the learned graph-based forecasting behaviour**.

📄 **Full report**: [Mini_Project_AST.pdf](./Mini_Project_AST.pdf)


---

## Datasets
- **SMAP**: 25 sensors (1 continuous, 24 binary), expert-labelled anomaly segments  
- **SMD**: 38 continuous sensors, single long multivariate series with anomaly segments  

Both datasets are loaded via `torch_timeseries`.  
Based on signal analysis, **no detrending or denoising** is applied.

---

## Method
We reuse the original **GDN architecture** and adapt all training, preprocessing, and evaluation code.

GDN:
- Learns **sensor embeddings**
- Builds a **Top-k similarity graph**
- Predicts the next time step using **graph attention**
- Detects anomalies via **normalized forecasting errors**

Training is done with **MSE loss**, as in the original paper.

---

## Experiments
- **Baselines**: VAR(100), AR(100)
- **Original GDN configuration** (`paper_config`)
- **Anomalies in training** (`anom_in_train`)
- **Context window size study** (SMAP vs SMD)
- **Point-wise and sequence-wise evaluation**

**SMAP only**:
- Mixed loss and score for continuous vs binary sensors (`mix_cont/bin`)

---

## Main Results
- **SMAP**: GDN performs poorly due to the binary nature of the data; AR(100) performs best.
- **SMD**: Higher AUC for all models; VAR(100) slightly outperforms GDN.
- Training with anomalies strongly degrades performance.
- Window size has opposite effects: small windows work best on SMAP, large windows on SMD.

---

## Code Reuse
We reuse the **GDN model architecture**; all other components are implemented by us.

---

## Run

```bash
jupyter notebook GDN.ipynb
```
---

## References
	•	Ailin Deng & Bryan Hooi — Graph Neural Network-Based Anomaly Detection in Multivariate Time Series (GDN), KDD 2021
	•	Franco Scarselli et al. — The Graph Neural Network Model, IEEE TNN, 2009
	•	NASA SMAP Anomaly Benchmark
	•	Server Machine Dataset (SMD) benchmark
