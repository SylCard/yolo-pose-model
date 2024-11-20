# YOLO Pose Training Project

This project implements YOLO pose model training with automatic checkpointing every 10 epochs on the COCO dataset.

## Requirements

- Python 3.7-3.11
- CUDA-capable GPU (for Windows/Linux) or M1/M2 Mac
- Git

## Setup Instructions

### For Windows/Linux with NVIDIA GPU:

1. Create and activate virtual environment:
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux
source venv/bin/activate
```

2. Install PyTorch with CUDA support:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

3. Install remaining requirements:
```bash
pip install -r requirements.txt
```

### For M1/M2 Mac:

1. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

2. Install PyTorch with MPS support:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
```

3. Install remaining requirements:
```bash
pip install -r requirements.txt
```

## Training

Run the training script:
```bash
python train_yolo_pose.py
```

The script will:
- Use the yolo11n-pose model
- Save checkpoints every 10 epochs in the 'checkpoints' directory
- Train on COCO pose dataset for 100 epochs
- Automatically use available GPU (CUDA for NVIDIA, MPS for M1/M2 Mac)

## Project Structure

- `train_yolo_pose.py`: Main training script
- `requirements.txt`: Project dependencies
- `.gitignore`: Git ignore configuration

## Notes

- For Windows/Linux: The script will automatically use CUDA if an NVIDIA GPU is available
- For M1/M2 Mac: The script will use Metal Performance Shaders (MPS) for GPU acceleration
- Checkpoints are saved every 10 epochs in the 'checkpoints' directory
- Virtual environment and checkpoints are excluded from git tracking

## Troubleshooting

If you encounter CUDA out of memory errors on Windows/Linux:
1. Reduce batch size in the training script
2. Reduce image size (imgsz parameter)
3. Ensure no other GPU-intensive tasks are running

For M1/M2 Mac memory issues:
1. Reduce batch size
2. Close unnecessary applications
3. Ensure sufficient swap space is available
