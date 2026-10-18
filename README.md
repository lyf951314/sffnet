# SFFNet

SFFNet is a PyTorch object detection network that combines spatial-domain and frequency-domain features. 

## 1\. Method overview

Based on the provided implementation, SFFNet contains:

* **SFA (Spatial-Domain Feature Awareness):** a ResNet-50-style encoder extracting multi-stage spatial features (`nets/sfa.py`).
* **FFA (Frequency-Domain Feature Awareness):** Haar discrete wavelet decomposition, high-frequency attention, and frequency-channel recalibration (`nets/ffa.py`).
* **SFI (Spatial-Frequency Interaction):** bidirectional local cross-attention between spatial and reconstructed frequency-domain features (`nets/sfi.py`).
* **Multi-scale fusion and U-shaped decoder:** combines features from several encoder stages (`nets/sffnet.py`).
* **Detection heads:** output a class heatmap, object size, and offset (`nets/sffnet.py`).

## 2\. Repository structure

```text
sffnet/
├── nets/
│   ├── ffa.py                     # Frequency-domain feature awareness
│   ├── sfa.py                     # Spatial-domain (ResNet-50) encoder
│   ├── sfi.py                     # Spatial-frequency interaction
│   ├── sffnet.py                  # Main detection architecture           
├── train_sffnet_ddp.py           
├── sffnet_inference.py
├── sffnet_pre.py                   # Prediction
├── evaluate_sffnet.py              # Performance Evaluation          
├── requirements.txt
└── README.md
```

## 3\. Software environment

The following versions are reported from the author's package list, **not** independently verified by a fresh installation.

|Component|Reported version|
|-|-|
|Python|3.10.16|
|PyTorch|1.13.1+cu116|
|Torchvision|0.14.1+cu116|
|Torchaudio|0.13.1+cu116|
|PyTorch CUDA wheel build|11.6|
|Pillow|12.3.0|
|SciPy|1.15.2|
|PyCOCOTools|2.0.8|
|TensorBoard|2.19.0|
|tqdm|4.65.2|

The `+cu116` suffix identifies the CUDA version of the installed PyTorch wheels. It does **not** establish the locally installed CUDA Toolkit (`nvcc`), NVIDIA driver version, GPU model, or operating-system release. Record these separately for exact hardware reproducibility.

For multi-GPU distributed training, the script uses the **NCCL** backend, so a CUDA-capable **Linux** environment is recommended. Uploading files from Windows does not change the training environment used for the experiments.

### Install

```bash
conda create -n sffnet python=3.10.16 -y
conda activate sffnet
python -m pip install -r requirements.txt
```

`requirements.txt` preserves the reported CUDA 11.6 PyTorch wheel versions. The supplied package-list excerpt did **not** specify versions for **NumPy**, **OpenCV**, or **Matplotlib**. Those three dependencies therefore use provisional compatibility constraints. Replace them with the actual installed versions from the experimental environment before calling the environment fully reproducible.

Check the active installation:

```bash
python -c "import sys, torch; print('Python:', sys.version); print('PyTorch:', torch.\_\_version\_\_); print('PyTorch CUDA:', torch.version.cuda); print('CUDA available:', torch.cuda.is\_available())"
python -m pip show numpy opencv-python matplotlib
```

