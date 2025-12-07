README — Graph Deviation Network (GDN) for Anomaly Detection on SMAP & SMD

This repository contains our implementation and experiments for the
MVA 2025/2026 — Machine Learning for Time Series mini-project.

We study the Graph Deviation Network (GDN) method introduced by Deng & Hooi (KDD 2021) and evaluate it on two multivariate time-series datasets: NASA SMAP and Server Machine Dataset (SMD).
Our work extends the original paper by analysing GDN under new experimental settings and on datasets not considered in the original study.

⸻

Dataset

We rely on two benchmark datasets, automatically downloaded through torch_timeseries:
	•	SMAP (NASA telemetry): 1 continuous sensor and 24 binary sensors
	•	SMD (Server Machine Dataset): 38 continuous sensors

Both datasets contain expert-labelled anomaly intervals.
No missing values are present, and we assume the training split contains only normal behaviour.

⸻

Model

We reuse the original GDN model architecture and adapt it to our data pipeline.
The model includes:
	•	learnable sensor embeddings
	•	Top-k graph construction using cosine similarity
	•	an attention-based graph forecasting module

We reused only the architecture and the training/testing routines, while all other components (data loading, preprocessing, evaluation, experiments, analysis tools) were fully implemented by us.

⸻

Additional Experiments

All experiments performed in this repository are new compared to the original GDN publication. They correspond to the analyses reported in our written report.

1. Data analysis and preprocessing
	•	SMAP: diagnosis of binary vs continuous sensors, autocorrelation, spectral analysis (DFT), detrending attempts, stationarity considerations
	•	SMD: examination of continuous sensors, regime-switch dynamics, autocorrelation study, spectral structure

2. Baselines and reproduction
	•	Reproduction of the original GDN configuration (paper_config)
	•	Implementation of linear baselines: AR(100) and VAR(100)

3. Mixed reconstruction loss for SMAP

To account for heterogeneous sensors:
	•	MSE on the continuous channel
	•	Binary Cross-Entropy on binary sensors
The anomaly scoring mechanism was adapted accordingly.

4. Robustness to anomalous training sets

We evaluate GDN when the training data is contaminated with anomalies, violating its normal-only assumption.

5. Influence of the context window size

GDN originally uses a fixed window size.
We evaluate window sizes w \in \{ 5, 10, 20, 50, 75, 100, 150, 200 \} on both datasets and analyse their impact on AUC.

⸻

How to Run

Open and execute:

gdn_experiments.ipynb

This notebook performs dataset loading, preprocessing, model training, scoring, evaluation, and visualization.

⸻

References
	•	Deng & Hooi — Graph Deviation Networks, KDD 2021
	•	Scarselli et al. — The Graph Neural Network Model, IEEE TNN 2009
	•	NASA Anomaly Benchmark (SMAP)
