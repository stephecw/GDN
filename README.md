# Graph Deviation Network (GDN) — Anomaly Detection on SMAP

This repository contains our implementation and experiments for the **MVA 2025/2026 — Machine Learning for Time Series** mini-project.
We study the **Graph Deviation Network (GDN)** method by Deng & Hooi (KDD 2021) and apply it to the **NASA SMAP** multivariate time-series dataset.

---

## Dataset

We use the **SMAP** dataset from the NASA telemetry anomaly benchmark.
It is automatically downloaded through:

---

## Model

We use the original **GDN architecture** and adapt it to our custom loaders.
The model learns:

* sensor embeddings
* a Top-k graph via cosine similarity
* an attention-based forecasting GNN

---

## Additional Experiments

Beyond reproducing the original GDN pipeline, we conduct several new experiments motivated by our analysis of the method:

* **data diagnosis** : missing values, stationarity, detrending, denoising

* **Robustness to anomalous training data**
  The original GDN assumes a fully normal training set. We test how sensitive the method is when anomalies contaminate the training split.

* **Alternative training loss**
  Since the MSE loss is highly sensitive to temporal misalignment, we experiment with replacing it by **soft-DTW**.

* **Varying the sliding-window size ( w )**
  The original paper fixes ( w = 5 ). We evaluate how different window lengths affect forecasting quality and anomaly detection performance.

* **Extreme Value Theory thresholding (POT)**
  Instead of the “max on validation” rule used in GDN, we apply a **Peaks-Over-Threshold (POT)** strategy to estimate anomaly thresholds in a statistically principled way and study its impact on score distribution and detection rates.

---

## How to Run

Simply run:

```
gdn_smap_paper.ipynb
```

which performs data loading, preprocessing, training, scoring, and visualisation.

---

## References

* Ailin Deng & Bryan Hooi — **Graph Deviation Networks**, KDD 2021
* Siffer et al. — **Anomaly Detection in Streams with EVT**, KDD 2017
* NASA SMAP Anomaly Benchmark
