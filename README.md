# Graph Convolutional Network (GCN) from Scratch

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A transparent, mathematical implementation of a 2-layer **Graph Convolutional Network (GCN)** built entirely from first principles in native PyTorch. This project contains **no dependencies on specialized GNN frameworks** (such as PyTorch Geometric or DGL), making every matrix multiplication, normalization trick, and message-passing step completely explicit.

The network performs **semi-supervised node classification** on **Zachary's Karate Club** ($N=34$ nodes), achieving $>85\text{--}90\%$ classification accuracy graph-wide while training on **only 2 labeled seed nodes** (Node 0 and Node 33).

---

## 📑 Table of Contents
- [Project Overview](#-project-overview)
- [Mathematical Foundations](#-mathematical-foundations)
  - [1. The Graph Laplacian](#1-the-graph-laplacian)
  - [2. The Renormalization Trick](#2-the-renormalization-trick)
  - [3. Spectral-to-Spatial Convolution](#3-spectral-to-spatial-convolution)
  - [4. Semi-Supervised Signal Propagation](#4-semi-supervised-signal-propagation)
- [Repository Structure](#-repository-structure)
- [🚀 Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Code](#running-the-code)
- [Expected Outputs & Visualizations](#-expected-outputs--visualizations)
- [Theoretical Roadmap](#-theoretical-roadmap)
- [References](#-references)
- [License](#-license)

---

## 📌 Project Overview

Traditional Machine Learning and Multi-Layer Perceptrons (MLPs) assume data instances are Independent and Identically Distributed ($i.i.d.$) across flat Euclidean space. On graphs, nodes exhibit structural homophily and non-Euclidean topology.

This project bridges spectral graph theory and practical deep learning:
* **No Graph Libraries:** Built purely on base `torch`, `networkx` (for dataset retrieval and layout), and `matplotlib`.
* **Complete Mathematical Transparency:** Explains how the unnormalized Laplacian $L = D - A$ leads directly to Kipf & Welling's renormalized symmetric propagation matrix $\hat{A}$.
* **Latent Manifold Inspection:** Uses a 2-dimensional hidden bottleneck ($H^{(1)} \in \mathbb{R}^{34 \times 2}$) to visualize how message passing clusters nodes linearly in Euclidean space before classification.

---

## 🧮 Mathematical Foundations

### 1. The Graph Laplacian
For an undirected graph with adjacency matrix $A \in \mathbb{R}^{N \times N}$ and diagonal degree matrix $D_{ii} = \sum_j A_{ij}$, the unnormalized Graph Laplacian is:

$$L = D - A$$

For any graph signal $x \in \mathbb{R}^N$, the quadratic form evaluates to:

$$x^T L x = \frac{1}{2} \sum_{i,j=1}^N A_{ij} (x_i - x_j)^2$$

This proves that $L$ is symmetric positive semi-definite ($x^T L x \ge 0$) and functions as a **smoothness penalty**: minimizing this term penalizes differences between connected neighbors.

### 2. The Renormalization Trick
Directly computing graph Fourier convolutions requires diagonalizing $L = U \Lambda U^T$, an $\mathcal{O}(N^3)$ operation. Kipf & Welling (2016) truncate a Chebyshev polynomial expansion to a 1st-order neighborhood. 

To prevent vanishing or exploding gradients caused by eigenvalues in $[0, 2]$, self-loops are added to the adjacency matrix before computing symmetric normalization:

1. **Add Self-Loops:**
   $$\tilde{A} = A + I_N$$
2. **Compute Augmented Degree:**
   $$\tilde{D}_{ii} = \sum_{j=1}^N \tilde{A}_{ij}$$
3. **Symmetric Normalization:**
   $$\hat{A} = \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}$$
   where each entry is:
   $$\hat{A}_{ij} = \frac{\tilde{A}_{ij}}{\sqrt{\tilde{d}_i \tilde{d}_j}}$$

### 3. Spectral-to-Spatial Convolution
Layer-wise forward propagation is defined as:

$$H^{(l+1)} = \sigma\left(\hat{A} H^{(l)} W^{(l)}\right)$$

Where:
* $H^{(0)} = X = I_{34}$ (one-hot node identities, since nodes lack initial scalar features).
* $W^{(l)}$ is a learnable parameter matrix.
* $\sigma(\cdot)$ is an activation function (ReLU for intermediate layers, linear/softmax for output logits).

### 4. Semi-Supervised Signal Propagation
Even though loss is calculated strictly on two nodes:

$$\mathcal{L} = \mathcal{L}(z_0, y_0) + \mathcal{L}(z_{33}, y_{33})$$

The entire network learns because:
* **Forward Pass:** The 2-hop operator $\hat{A}^2$ diffuses features across paths of length 2, blending neighbor representations.
* **Backward Pass:** The prediction $z_0$ depends on $\sum_{j \in \mathcal{N}(0)} h_j^{(1)}$. Backpropagation routes gradients through the hidden vectors of unlabeled neighbors to update shared weights $W$.

---

## 📂 Repository Structure

```text
.
├── gcn_karate_club.py     # End-to-end Python script (Data, Model, Visualizations)
├── requirements.txt       # Minimal dependency requirements
├── LICENSE                # MIT License
└── README.md              # Technical documentation & usage instructions
```

---

## 🚀 Getting Started

Follow these steps to set up and run the implementation locally.

### Prerequisites
* Python 3.9 or higher
* `pip` package manager

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/gcn-from-scratch.git
   cd gcn-from-scratch
   ```

2. **(Optional) Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # On macOS/Linux:
   source venv/bin/activate
   # On Windows:
   venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Code

Execute the pipeline with a single command:

```bash
python gcn_karate_club.py
```

---

## 📊 Expected Outputs & Visualizations

When executed, the script runs the complete training loop, logs metrics, and generates visual plots:

### 1. Terminal Output
```text
Training GCN on Karate Club...
Epoch   1 | Train Loss: 0.6931 | Graph-wide Accuracy: 50.0%
Epoch  30 | Train Loss: 0.4120 | Graph-wide Accuracy: 76.5%
Epoch  60 | Train Loss: 0.1845 | Graph-wide Accuracy: 85.3%
Epoch  90 | Train Loss: 0.0892 | Graph-wide Accuracy: 88.2%
Epoch 120 | Train Loss: 0.0461 | Graph-wide Accuracy: 91.2%
Epoch 150 | Train Loss: 0.0274 | Graph-wide Accuracy: 91.2%

--- Model Output Summary ---
Total Nodes Classified: 34
Correct Predictions:    31 / 34
Overall Accuracy:       91.18%
```

### 2. Generated Plots
The script opens two visualization windows:
1. **Input Stage:** The ground-truth graph topology alongside the semi-supervised training mask (where only Nodes 0 and 33 are highlighted).
2. **Evaluation Stage:** 
   * **Left:** Final GCN classifications on the graph topology with boundary misclassifications highlighted.
   * **Right:** The 2D latent space coordinates ($H^{(1)}$), showing the physical separation between Mr. Hi's and the Officer's factions prior to final classification.

---

## 🧭 Theoretical Roadmap

This project forms the baseline for understanding modern graph architectures:

| Architecture | Paradigm | Key Change from GCN |
| :--- | :--- | :--- |
| **Vanilla GCN** (Kipf & Welling, 2016) | Spectral / Transductive | Baseline: full-matrix $\hat{A} H W$ with $\text{ReLU}$. |
| **LightGCN** (He et al., 2020) | Linear / Recommendation | Strips out feature transformation $W$ and non-linearities $\sigma$, using purely normalized graph diffusion. |
| **GraphSAGE** (Hamilton et al., 2017) | Spatial / Inductive | Replaces full-graph matrix normalization with uniform neighborhood sampling and modular aggregation functions (Mean, Pool, LSTM). |

---

## 📚 References

1. **Kipf, T. N., & Welling, M. (2016).** *Semi-Supervised Classification with Graph Convolutional Networks.* [arXiv:1609.02907](https://arxiv.org/abs/1609.02907).
2. **Zachary, W. W. (1977).** *An Information Flow Model for Conflict and Fission in Small Groups.* Journal of Anthropological Research, 33(4), 452–473.
3. **Defferrard, M., Bresson, X., & Vandergheynst, P. (2016).** *Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering.* NeurIPS 2016.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.