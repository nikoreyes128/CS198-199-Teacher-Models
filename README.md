# CS 198 / CS 199 Thesis: Phase 0 – Teacher Models Baseline Evaluation

This repository contains the evaluation code, benchmarking pipelines, and configuration for **Phase 0** of the thesis project. The goal of this phase is to establish baseline performance metrics across candidate teacher object detection models on low-light imagery using the **ExDark (Exclusively Dark)** dataset.

Candidate architectures evaluated:
- **YOLOv8s** (Ultralytics)
- **RT-DETR-R50vd** (Real-Time DEtection TRansformer / Peking University & Baidu)
- **Gold-YOLO** (Huawei Noah's Ark Lab / Efficient-Computing)

---

## Table of Contents
1. [Repository Structure](#repository-structure)
2. [Prerequisites](#prerequisites)
3. [Environment Setup](#environment-setup)
   - [Unified Virtual Environment (Recommended)](#option-a-unified-virtual-environment-recommended)
   - [PyTorch with GPU / CUDA Acceleration](#gpu--cuda-installation-recommended)
4. [Dataset Setup (ExDark)](#dataset-setup-exdark)
5. [Running the Baselines](#running-the-baselines)
   - [1. YOLOv8s Baseline](#1-yolov8s-baseline)
   - [2. RT-DETR-R50vd Baseline](#2-rt-detr-r50vd-baseline)
   - [3. Gold-YOLO Baseline](#3-gold-yolo-setup)
6. [Evaluation Metrics & Methodology](#evaluation-metrics--methodology)
7. [Troubleshooting & FAQs](#troubleshooting--faqs)

---

## Repository Structure

```text
phase-0/
├── dataset/                    # Dataset directory (ignored by git)
│   └── ExDark-coco/
│       ├── valid/
│       │   ├── _annotations.coco.json
│       │   └── *.jpg
│       ├── train/
│       └── test/
├── gold-yolo/                  # Gold-YOLO setup & scripts
│   ├── .gitkeep
│   └── Efficient-Computing/    # Cloned Gold-YOLO upstream repo (ignored by git)
├── results/                    # Output directory for summary reports (ignored by git)
│   └── .gitkeep
├── rt-detr/                    # RT-DETR evaluation scripts
│   ├── rtdetr-test.py          # RT-DETR evaluation & profiling script
│   └── test.py                 # Ground truth inspection helper
├── yolov8/                     # YOLOv8 evaluation scripts
│   └── yolov8-test.py          # YOLOv8s evaluation & profiling script
├── .gitignore                  # Git ignore rules
├── README.md                   # This setup guide
└── requirements.txt            # Unified project Python dependencies
```

> **Note:** The `dataset/` directory, model checkpoint files (`*.pt`, `*.bin`, `*.safetensors`), virtual environments (`env-*/`), and generated prediction outputs (`*predictions*.json`) are excluded from Git to prevent repository bloat and comply with GitHub's file size restrictions.

---

## Prerequisites

- **OS**: Windows 10/11, Ubuntu 20.04+, or macOS (Apple Silicon / Intel)
- **Python**: Version `3.10` to `3.12` (tested with Python 3.12)
- **CUDA Toolkit** (Optional but strongly recommended): CUDA 11.8 or 12.1+ for NVIDIA GPU inference and accurate FPS benchmarking
- **Git**: Installed and available in your command line path

---

## Environment Setup

### Option A: Unified Virtual Environment (Recommended)

You can run both YOLOv8 and RT-DETR within a single virtual environment.

#### Windows (PowerShell)
```powershell
# 1. Clone the repository
git clone https://github.com/nikoreyes128/CS198-199-Teacher-Models.git
cd CS198-199-Teacher-Models

# 2. Create a virtual environment
python -m venv venv

# 3. Activate the virtual environment
.\venv\Scripts\Activate.ps1

# 4. Upgrade pip and install wheel
python -m pip install --upgrade pip setuptools wheel

# 5. Install the dependencies
pip install -r requirements.txt
```

#### Linux / macOS (bash / zsh)
```bash
# 1. Clone the repository
git clone https://github.com/nikoreyes128/CS198-199-Teacher-Models.git
cd CS198-199-Teacher-Models

# 2. Create a virtual environment
python3 -m venv venv

# 3. Activate the virtual environment
source venv/bin/activate

# 4. Upgrade pip and install wheel
python -m pip install --upgrade pip setuptools wheel

# 5. Install the dependencies
pip install -r requirements.txt
```

### GPU / CUDA Installation (Recommended)

If you have an NVIDIA GPU, ensure PyTorch is installed with CUDA support for hardware-accelerated evaluation:

```bash
# Example for CUDA 12.1 (adjust according to your CUDA driver)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

To verify your GPU is recognized:
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

## Dataset Setup (ExDark)

The evaluation scripts are designed to validate predictions against the **Exclusively Dark (ExDark)** dataset in **COCO JSON format**.

1. Download or export the ExDark COCO dataset (e.g., from [Roboflow Universe: ExDark](https://app.roboflow.com/thomas-nikolas-reyes/exdark-kd37x-ybab5/1)).
2. Create the `dataset/ExDark-coco/` folder in the project root and extract the validation split so that the annotations and images match this path structure:

```text
phase-0/
└── dataset/
    └── ExDark-coco/
        └── valid/
            ├── _annotations.coco.json
            ├── 2015_00010_jpg.rf....jpg
            ├── 2015_00027_jpg.rf....jpg
            └── ... (all valid images)
```

### ExDark Class Mapping
ExDark consists of 12 target classes:
1. `Bicycle` (1)
2. `Boat` (2)
3. `Bottle` (3)
4. `Bus` (4)
5. `Cat` (5)
6. `Cup` (6)
7. `Motorbike` / `motorcycle` (7)
8. `People` / `person` (8)
9. `Table` / `dining table` (9)
10. `car` (10)
11. `chair` (11)
12. `dog` (12)

The evaluation scripts automatically filter and remap the 80 COCO pre-trained categories into these 12 ExDark category IDs.

---

## Running the Baselines

> **Important:** Always change your directory into the respective model folder before executing the test script, as relative dataset paths (`../dataset/ExDark-coco/valid/`) originate from there.

### 1. YOLOv8s Baseline

The script initializes `yolov8s.pt` (automatically downloaded by Ultralytics if not found locally), runs batchless inference over the ExDark validation set, maps prediction IDs, and computes COCO evaluation metrics alongside parameter count and FLOPs.

```bash
# Navigate to the yolov8 directory
cd yolov8

# Run evaluation
python yolov8-test.py
```

**Expected Outputs:**
- Model Parameters & FLOPs printed to the terminal.
- Total inference time and Average FPS.
- Standard COCO summary metrics (mAP@0.50:0.95, mAP@0.50, etc.).
- A local file `yolo_mapped_predictions.json` (auto-ignored by git).

### 2. RT-DETR-R50vd Baseline

The script loads the `PekingU/rtdetr_r50vd` model and image processor directly from Hugging Face Transformers, measures hardware complexity with `fvcore`, runs inference, and computes COCO metrics.

```bash
# Navigate to the rt-detr directory
cd rt-detr

# Run evaluation
python rtdetr-test.py
```

**Expected Outputs:**
- Model Parameters (Million) and Computational Cost (GFLOPs).
- Total inference time and Average FPS.
- Standard COCO summary metrics (mAP@0.50:0.95, mAP@0.50).
- A local file `rtdetr_predictions.json` (auto-ignored by git).

### 3. Gold-YOLO Setup

Gold-YOLO relies on Huawei Noah's Ark Lab's `Efficient-Computing` repository. To set it up:

```bash
# Navigate to the gold-yolo directory
cd gold-yolo

# Clone the official upstream repository
git clone https://github.com/huawei-noah/Efficient-Computing.git

# Navigate into the Gold-YOLO project
cd Efficient-Computing/Detection/Gold-YOLO

# Install any specialized dependencies (e.g. mmcv, addict)
pip install -r requirements.txt
```

---

## Evaluation Metrics & Methodology

Each baseline script collects two categories of metrics:

### 1. Hardware & Computational Efficiency
- **Parameter Count ($M$)**: Total learnable model parameters.
- **Computational Complexity ($\text{GFLOPs}$)**: Floating-point operations required for a single $640 \times 640$ forward pass (calculated using `fvcore` or Ultralytics model profiler).
- **Latency ($\text{ms}$)**: Pure forward-pass inference latency per image.
- **Throughput ($\text{FPS}$)**: Total images divided by total pure inference time ($N / \sum t_{\text{inference}}$).

### 2. Detection Performance (COCO API)
Evaluated via `pycocotools.cocoeval.COCOeval`:
- **$\text{mAP@[0.50:0.95]}$**: Primary COCO object detection metric averaged over IoU thresholds from 0.50 to 0.95 with step size 0.05.
- **$\text{mAP@0.50}$**: PASCAL VOC-style metric at IoU threshold 0.50.
- **$\text{mAP@0.75}$**: Strict detection threshold at IoU 0.75.
- **$\text{mAP}_{\text{small}}$, $\text{mAP}_{\text{medium}}$, $\text{mAP}_{\text{large}}$**: Scale-specific detection metrics.

---

## Troubleshooting & FAQs

### 1. `FileNotFoundError: [Errno 2] No such file or directory: '../dataset/ExDark-coco/valid/_annotations.coco.json'`
- **Cause**: Running the script from the root directory instead of the model folder, or missing dataset files.
- **Solution**:
  - Ensure you `cd yolov8` or `cd rt-detr` before executing the script.
  - Verify that `_annotations.coco.json` exists in `phase-0/dataset/ExDark-coco/valid/`.

### 2. `CUDA out of memory`
- **Solution**: The evaluation loops run single-image inference by default (`batch size = 1`). If GPU VRAM is constrained, you can force CPU execution by setting `device = torch.device('cpu')` in the scripts.

### 3. Windows PowerShell Execution Policy Error on `activate.ps1`
- **Cause**: PowerShell restricts running unsigned scripts by default.
- **Solution**: Run this command once in PowerShell:
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
  ```

