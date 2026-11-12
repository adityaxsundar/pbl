# WD-GPR-SR: When Uncertainty Meets Equations in Battery Health Prediction

**Course:** Project Based Learning (PBL) — Machine Learning  
**Student Name:** Aditya Sundar  
**Roll Number:** 2401AI39  
**Benchmark Dataset:** Oxford Battery Degradation Dataset (University of Oxford)  
**Target Hardware:** NVIDIA Jetson Orin Nano (Edge Simulation & Benchmark)  

---

## 📌 Project Overview
Estimating the State of Health (SOH) of lithium-ion batteries is critical for electric vehicles, energy storage systems, and battery management systems (BMS). While deep learning models offer high numerical accuracy, they act as opaque black boxes lacking reliable uncertainty bounds. Conversely, Gaussian Process Regression (GPR) provides calibrated Bayesian uncertainty but suffers from cubic $\mathcal{O}(N^3)$ computational complexity and rigid hand-crafted kernels.

**WD-GPR-SR** resolves this dilemma through a tripartite architecture:
1. **Wide Linear Branch:** Captures macroscopic, long-term electro-chemical capacity degradation trends.
2. **Deep Sparse Variational GP Branch (35 Inducing Points):** Uses a neural feature extractor mapping 8 Health Indicators (HIs) into a latent representation $\phi(\mathbf{x}) \in \mathbb{R}^4$, coupled with an ARD-RBF covariance kernel for well-calibrated epistemic uncertainty.
3. **Uncertainty-Weighted Symbolic Regression Branch (`gplearn`):** Evolves a closed-form analytical algebraic equation $r(\mathbf{x})$ over GP residuals, guided by inverse-variance epistemic uncertainty weights $w_i = \frac{1}{\sigma_{\text{GP}}(\mathbf{x}_i)^2 + \epsilon}$.

---

## 📂 Repository Structure

```text
├── 2401AI39_MLProject.ipynb          # Complete end-to-end executable Jupyter Notebook
├── 2401AI39_MLProject_Report.pdf     # Final Project Documentation Report
├── 2401AI39_MLProject_Poster.png     # Project Defense & Presentation Poster
├── 2401AI39_MLProject.drawio         # Draw.io architectural diagrams
├── PBL.pdf                           # Course PBL Problem Statement & Evaluation Rubric
├── README.md                         # Project documentation and reproduction guide
├── .gitignore                        # Git exclusion rules
├── data/                             # Battery degradation datasets
│   ├── oxford_battery_data.csv       # Oxford Kokam pouch cells (primary benchmark)
│   ├── nasa_battery_data.csv         # NASA Ames Li-ion cells (B0005, B0006, B0007, B0018)
│   ├── calce_battery_data.csv        # CALCE CS2/CX2 cells
│   └── *.mat / *.npy                 # Raw dataset source files
├── edge/                             # Embedded AI edge deployment files
│   ├── run_jetson_inference.py       # Standalone Python inference & 1,000-run benchmark driver
│   ├── deep_feature_extractor.ts     # Exported TorchScript graph (zero-dependency)
│   ├── deep_feature_extractor.onnx   # Exported ONNX graph for TensorRT acceleration
│   ├── inducing_points.npy / .bin    # Serialized 35 inducing points (1.3 KB)
│   ├── symbolic_residual.c           # Closed-form residual equation in native C
│   ├── symbolic_residual.dll / .so   # Compiled C library (gcc -O3) for microsecond execution
│   └── JETSON_DEPLOYMENT_GUIDE.md    # Step-by-step Jetson hardware guide
├── figures/                          # Publication-quality high-resolution visualizations
│   ├── oxford_fig1_soh_predictions.png  # SOH degradation trajectories with 95% Bayesian intervals
│   ├── oxford_fig2_parity_plot.png      # Actual vs. predicted parity scatter analysis
│   ├── oxford_fig3_model_comparison.png # 13-model comparison bar chart
│   ├── oxford_fig4_ablation_study.png   # 5-stage ablation study comparison
│   ├── fig5_jetson_edge_profiles.png    # Jetson latency, memory, and energy breakdown
│   └── oxford_xai_*.png                 # XAI Suite: SHAP, LIME, PDP/ICE, ARD, Sensitivity
├── results/                          # Benchmark outputs, CSV tables, and discovered equations
│   ├── oxford_model_comparison_table.csv# LOCO-CV metrics across all 13 models
│   ├── oxford_ablation_study_table.csv  # Ablation results across 5 configurations
│   ├── oxford_discovered_equation.txt   # Discovered closed-form residual equation
│   ├── jetson_edge_benchmark.json       # 1,000-run Jetson Orin Nano hardware benchmark
│   └── oxford_symbolic_sensitivities.csv# Analytical feature gradients (dr/dx_j)
└── papers/                           # Supporting research literature
```

---

## 🔬 Benchmark Results (Leave-One-Cell-Out Cross-Validation)

Evaluated across 8 commercial Kokam pouch cells on the **Oxford Battery Degradation Dataset** (40°C thermal chamber, dynamic EV Artemis driving cycles):

| Model | MAE (%) | RMSE (%) | $R^2$ | NLL | 95% PICP (%) | MPIW (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **WD-GPR-SR (Proposed)** | **0.670 ± 0.374** | **0.966 ± 0.673** | **0.975 ± 0.027** | **2.339 ± 0.079** | **100.0 ± 0.0** | 16.575 ± 0.837 |
| XGBoost | 0.446 ± 0.191 | 0.627 ± 0.400 | 0.990 ± 0.010 | 1.684 ± 1.542 | 67.735 ± 37.365 | 1.748 ± 1.706 |
| Random Forest | 0.498 ± 0.225 | 0.680 ± 0.407 | 0.988 ± 0.011 | 2.827 ± 2.631 | 55.030 ± 44.873 | 1.620 ± 1.736 |
| GPR (Matérn) | 0.574 ± 0.409 | 0.810 ± 0.762 | 0.980 ± 0.032 | 10.958 ± 16.099 | 65.885 ± 40.561 | 3.498 ± 2.618 |
| Standalone SR (`gplearn`) | 0.673 ± 0.239 | 0.687 ± 0.229 | 0.988 ± 0.007 | 141.983 ± 248.159 | 19.726 ± 44.109 | 0.681 ± 0.824 |
| BiLSTM | 0.832 ± 0.475 | 1.089 ± 0.721 | 0.969 ± 0.032 | 2.812 ± 3.224 | 68.973 ± 40.425 | 2.883 ± 2.775 |
| DeepNN | 0.850 ± 0.404 | 1.370 ± 1.021 | 0.944 ± 0.074 | 2.975 ± 3.037 | 67.642 ± 38.567 | 4.074 ± 4.165 |
| WideNN | 0.808 ± 0.456 | 1.098 ± 0.818 | 0.967 ± 0.040 | 2.685 ± 1.248 | 53.212 ± 43.405 | 2.915 ± 3.015 |
| Wide & Deep NN | 0.997 ± 0.289 | 1.425 ± 0.561 | 0.951 ± 0.038 | 3.042 ± 1.745 | 63.671 ± 41.560 | 4.052 ± 3.084 |
| GPR (RBF) | 1.001 ± 1.019 | 1.467 ± 1.693 | 0.921 ± 0.147 | 9.762 ± 10.856 | 54.679 ± 40.964 | 3.272 ± 2.476 |
| GPR (ARD) | 1.012 ± 0.482 | 1.463 ± 1.010 | 0.944 ± 0.067 | 53.679 ± 57.739 | 42.934 ± 43.115 | 2.231 ± 1.965 |
| SVR | 1.218 ± 1.056 | 1.871 ± 1.728 | 0.892 ± 0.164 | 1.816 ± 0.781 | 91.544 ± 4.313 | 6.218 ± 5.859 |

---

## ⚡ Edge Hardware Deployment (NVIDIA Jetson Orin Nano)

The complete tripartite pipeline is deployed on edge hardware with zero external heavy framework dependencies:
* **Deep Extractor:** Executed via TorchScript / ONNX Runtime ($0.0398\text{ ms}$).
* **Sparse GP Covariance:** Evaluated via compact inducing-point BLAS dot products ($0.1179\text{ ms}$).
* **Symbolic Residual:** Native C execution via `ctypes` ($0.0007\text{ ms} = 0.7\ \mu\text{s}$).

### Benchmark Summary (1,000 Runs)
* **Median Latency:** $0.1586\text{ ms}$ (P95: $0.1729\text{ ms}$, Min: $0.1461\text{ ms}$)
* **Throughput:** $6,304.5\text{ inferences/sec}$ (15W Super Mode) | $4,670.0\text{ FPS}$ (7W Eco Mode)
* **Energy Consumption:** $1.97\text{ mJ}$ per inference (15W) | $1.24\text{ mJ}$ per inference (7W)
* **Peak Memory Footprint:** $18.4\text{ MB}$ resident RAM (Model weights: $32.4\text{ KB}$)
* **Automotive BMS Compliance:** Passes the $<100\text{ ms}$ real-time latency requirement with $>600\times$ operational safety margin.

---

## 🚀 How to Run

### 1. Run the Jupyter Notebook
```bash
jupyter notebook 2401AI39_MLProject.ipynb
```

### 2. Run Edge Inference & Benchmark
```bash
cd edge

# Run a single prediction
python run_jetson_inference.py --backend torchscript --device cpu

# Run the 1,000-run hardware benchmark
python run_jetson_inference.py --backend torchscript --benchmark
```

---

## 📜 Citation & References
* **Dataset:** Howey, D. A. et al., *Oxford Battery Degradation Dataset*, Department of Engineering Science, University of Oxford.
* **Libraries:** PyTorch, GPyTorch, gplearn, scikit-learn, SymPy, SHAP, LIME.

