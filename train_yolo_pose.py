from ultralytics import YOLO
import os
import torch

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def train_with_checkpoints():
    # Create checkpoints directory if it doesn't exist
    checkpoint_dir = "checkpoints"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Load a pretrained model
    model = YOLO("yolo11n-pose.pt")  # Using yolo11n-pose model as specified

    # Get appropriate device
    device = get_device()
    print(f"Using device: {device}")

    # Define training arguments
    training_args = {
        "data": "coco-pose.yaml",  # COCO pose dataset configuration
        "epochs": 100,             # Total number of epochs
        "imgsz": 640,             # Input image size
        "save_period": 10,         # Save checkpoint every 10 epochs
        "project": checkpoint_dir, # Save results to checkpoints directory
        "name": "train",          # Name of the training run
        "exist_ok": True,         # Overwrite existing files
        "pretrained": True,       # Use pretrained weights
        "verbose": True,          # Print verbose output
        "device": device          # Use appropriate device (cuda/mps/cpu)
    }

    # Train the model with the specified arguments
    try:
        print(f"Available device types:")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"MPS available: {hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()}")
        results = model.train(**training_args)
        print("Training completed successfully!")
        print(f"Checkpoints saved in: {os.path.abspath(checkpoint_dir)}")
        return results
    except Exception as e:
        print(f"An error occurred during training: {str(e)}")
        return None

if __name__ == "__main__":
    print("Starting YOLO Pose training with checkpoints...")
    train_with_checkpoints()
