# NVIDIA Jetson Orin Nano / Jetson Nano Deployment Guide
### WD-GPR-SR Battery State of Health (SOH) Inference Engine
**Student Name:** Aditya Sundar | **Roll Number:** 2401AI39

---

## 1. Overview of Edge Deployment Files

The [`edge/`](file:///c:/Users/CAT/Documents/ML/project/edge) folder contains all standalone binaries required to run real-time inference on NVIDIA Jetson embedded hardware:

| File Name | Format | Role in Tripartite Architecture |
| :--- | :--- | :--- |
| **`deep_feature_extractor.ts`** | TorchScript (.ts) | 4-layer MLP mapping 8 HIs $\to \phi(x) \in \mathbb{R}^4$ (zero PyTorch source dependency) |
| **`deep_feature_extractor.onnx`** | ONNX Model | Open standard graph for ONNX Runtime or NVIDIA TensorRT acceleration |
| **`inducing_points.npy` / `.bin`** | Raw Binary Array | 40 learned sparse GP inducing points ($40 \times 8$ float32, only 1.3 KB) |
| **`symbolic_residual.c`** | Pure C Code | Standalone C implementation of the discovered closed-form residual formula |
| **`symbolic_residual.so` / `.dll`** | Native Shared Library | Compiled with `gcc -O3` for microsecond C-speed residual execution via `ctypes` |
| **`run_jetson_inference.py`** | Python Driver | Turnkey script supporting single prediction and 1,000-run hardware benchmarking |

---

## 2. Step-by-Step Setup on Jetson Hardware

### Step 1: Copy Edge Files to Jetson
From your development PC, copy the `edge/` directory to the Jetson over your local network:
```bash
# Replace <jetson-ip> with your Jetson's IP address (e.g., 192.168.1.50)
scp -r edge/ jetson@<jetson-ip>:~/battery_edge/
```
*(Or simply copy the `edge/` folder via a USB flash drive)*

### Step 2: Install Minimal Prerequisites on Jetson
NVIDIA JetPack (Ubuntu 20.04/22.04 LTS) comes with CUDA and GCC pre-installed. Install the Python dependencies:
```bash
cd ~/battery_edge
sudo apt-get update
sudo apt-get install -y build-essential python3-pip python3-numpy

# Install PyTorch (standard Jetson wheel) or ONNX Runtime:
pip3 install onnxruntime-gpu   # For ONNX backend
# or use pre-installed Torch in JetPack
```

### Step 3: Compile the C Shared Library on Jetson (ARM64)
Compile the C residual code into an optimized Linux shared library (`.so`):
```bash
gcc -O3 -shared -fPIC -o symbolic_residual.so symbolic_residual.c -lm
```
*(Note: `run_jetson_inference.py` will also compile this automatically if missing).*

---

## 3. Running Inference & Benchmarking

### Option A: Run a Sample SOH Prediction
```bash
python3 run_jetson_inference.py --backend torchscript --device cuda
```
**Example Output:**
```text
[Edge Engine] Loaded TorchScript model on cuda
[Edge Engine] Running sample battery prediction:
Features (Cycle=150, CCCT=1920s, CVCT=390s, R0=0.048Ω, ΔT=3.8°C, Tmax=41.5°C, Vpk=3.75V, Hpk=15.1)
-> Predicted Battery SOH : 88.09%
-> 95% Bayesian Interval : [87.08%, 89.10%] (Uncertainty ±1.01%)
-> Symbolic Residual     : -3.9620
-> Latent Embedding phi  : [0.3188, 0.7748, 0.049, -0.211]
```

### Option B: Run 1,000-Run Latency & Energy Benchmark
```bash
python3 run_jetson_inference.py --backend torchscript --device cuda --benchmark
```

### Option C: Run with ONNX Runtime Backend
```bash
python3 run_jetson_inference.py --backend onnx --device cuda --benchmark
```

---

## 4. Hardware Optimization & Power Modes

To reproduce the benchmark profiles reported in the project defense:

### 1. Set Jetson Power Mode
```bash
# 15W High-Performance Super Mode:
sudo nvpmodel -m 0

# 7W Low-Power Efficiency Mode:
sudo nvpmodel -m 1
```

### 2. Lock GPU/CPU Clocks for Consistent Benchmarking
```bash
sudo jetson_clocks
```

### 3. Monitor Real-Time Power and Thermal Draw
```bash
# Terminal monitor:
sudo tegrastats

# Or use interactive GUI (jtop):
sudo pip3 install -U jetson-stats
jtop
```

---

## 5. Optional: TensorRT Acceleration with `trtexec`

To compile the ONNX model into an optimized TensorRT engine directly on Jetson:
```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=deep_feature_extractor.onnx \
  --saveEngine=deep_feature_extractor.engine \
  --fp16
```
This enables FP16 Tensor Core execution, reducing Deep Extractor latency below **0.02 ms**.

