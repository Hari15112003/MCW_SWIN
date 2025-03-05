

---

# Model Training and Quantization  

## Overview  
This project involves training a deep learning model SWIN with quantization techniques for improved efficiency. The configuration file defines parameters for training, quantization, and dataset paths.  

## Training Details  
- **Device**: CUDA (GPU)  
- **Number of Classes**: 1000  
- **Batch Size**: 32 (training), 128 (general)  
- **Image Size**: 224x224  
- **Pretrained Model**: Enabled  
- **Epochs**: 10 (max 20)  
- **Optimizer**: Adam  
- **Learning Rate**: 5e-7 (decays at epochs 5 & 10)  
- **Weight Decay**: 1e-4  

## Quantization Settings  
- **Quantization Type**: QAT & PTQ  
- **Bit Width**: 8-bit (weights & activations)  
- **Auto Quantization**: Supported (default 2000 iterations, 1% accuracy drop allowed)  
- **Post Training Quantization (PTQ)**: CLE enabled  

## Performance Results  

| Approach     | Top-1 Accuracy | Top-5 Accuracy |  
|-------------|--------------|--------------|  
| **Baseline** | 81.52%       | 95.60%       |  
| **Manual QAT** | 79.54%       | 94.74%       |  
| **PTQ - CLE** | 89.52%       | 98.80%       |  
| **AutoQuant** | 91.32%       | 99.17%       |  
| **QAT**       | 89.56%       | 98.94%       |  

## Paths  
- **Model Save Path**: `/media/ava/workspace/harish/swin/artifacts`  
- **Log File**: `/media/ava/workspace/harish/swin/logs/app.log`  
- **Dataset**:  
  - **Train**: `/media/ava/DATA/datasets/50k_imagenet/train/`  
  - **Validation**: `/media/ava/DATA/datasets/50k_imagenet/val/`  

## Logging  
- **Log Level**: Info  
