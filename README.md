<h1 align="center">
Back to the Future: Look-ahead Augmentation and Parallel Self-Refinement for Time Series Forecasting
</h1>

<p align="center">
  <img src="https://img.shields.io/badge/WWW'26-Short%20Paper-ff8c00"/>
</p>

Official implementation of **Back to the Future (BTTF)**, a simple yet effective framework for **long-term time series forecasting (LTSF)** via **look-ahead augmentation**, **parallel self-refinement**, and **Ensembled Forecasting**.

BTTF leverages segments of model-generated future predictions as **future-aware context**, enabling DMS-style parallelism while implicitly retaining IMS-like temporal dependency.

<p align="center">
  <img src="image/fig1.jpg" width="60%"/>
</p>
<p align="center"><em>
</em></p>


### Method Overview

BTTF is a two-stage framework that refines a base forecasting model using **future-aware augmentation** and **parallel self-refinement**, then stabilizes predictions with a **step-wise top-K ensemble**.

#### 1) Look-ahead Augmentation
Given a first-stage forecast, split the predicted horizon into **N segments** and append each segment to the original input window to form augmented inputs.

#### 2) Parallel Self-Refinement
Train **N independent** second-stage predictors on the augmented inputs; each model learns a distinct refinement pattern.

#### 3) Ensembled Forecasting
Rank second-stage models by validation performance and compute a **step-wise top-K** ensemble; choose the optimal $\(K^*\)$ via variance and co-variance analysis.

<p align="center">
  <img src="image/fig2.jpg" width="100%"/>
</p>
<p align="center"><em>
</em></p>



# Quick Start


# Citation






