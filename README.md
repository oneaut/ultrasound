# Ultrasound Imaging for Gait Analysis

Scripts to beamform raw PSRF ultrasound data, extract image‐based features (speckle‐tracking, strain, intensity), and perform gait analysis on unassisted treadmill walking trials. Supports both CPU and GPU workflows and includes US processing pipelines for real‑time and offline analysis.

---

## Repository Structure

.ultrasound/
├── beamforming_GPU.m # GPU‑accelerated planewave beamformer

├── planewaveBeamform.m # Core MATLAB planewave beamforming function

├── unassisted_gait_analysis.m # MATLAB script: end‑to‑end B‑mode → feature extraction → plots

├── us_psrf.py # Python reader for raw PSRF files

├── us_processing.py # Python beamforming & envelope → JPEGs & MP4

├── us_processing_robust_gait.py # Python: feature extraction 



---

## Features

- **CPU & GPU beamforming** in MATLAB (`planewaveBeamform.m` & `beamforming_GPU.m`)  
- **Raw PSRF file reading** in Python (`us_psrf.py`)  
- **B‑mode image generation** (Hilbert envelope + log compression)  
- **Speckle‑tracking & strain estimation** for muscle displacement  
- **gait analysis** phase detections (MATLAB & Python)  

---

## Requirements

- **MATLAB** R2018b or later  
  - Signal Processing Toolbox (for `hilbert`)  
  - Parallel Computing Toolbox (optional GPU)  
- **Python** 3.7+  
  - `numpy`  
  - `scipy`  
  - `matplotlib`  
  - `opencv-python`  
  - `scikit-learn`

---

## Installation

1. **Clone the repo**  
   ```bash
   git clone https://github.com/oneaut/ultrasound.git
   cd ultrasound

