from ultralytics import YOLO
import os
import glob

def get_latest_model():
    # Look for the latest model in checkpoints/train/weights
    weight_path = "checkpoints/train/weights"
    if not os.path.exists(weight_path):
        print("No trained weights found. Using pretrained model.")
        return "yolo11n-pose.pt"
    
    # Get all .pt files in the weights directory
    weights = glob.glob(os.path.join(weight_path, "*.pt"))
    if not weights:
        print("No trained weights found. Using pretrained model.")
        return "yolo11n-pose.pt"
    
    # Get the latest weight file
    latest_weight = max(weights, key=os.path.getctime)
    print(f"Using latest trained model: {latest_weight}")
    return latest_weight

def run_prediction():
    # Check if test video exists
    if not os.path.exists("test.mov"):
        print("Error: test.mov not found in current directory")
        return
    
    # Get the latest model
    model_path = get_latest_model()
    
    try:
        # Load the model
        model = YOLO(model_path)
        
        # Run prediction on video
        print("Starting prediction on test.mov...")
        results = model.predict(
            source="test.mov",
            save=True,           # Save results
            project="predictions",# Save to predictions folder
            name="test_output",  # Subfolder name
            conf=0.5,           # Confidence threshold
            show=True,          # Display prediction
            device=None         # Auto-select device (will use same as training)
        )
        
        print(f"Prediction completed. Results saved in: {os.path.abspath('predictions/test_output')}")
        
    except Exception as e:
        print(f"An error occurred during prediction: {str(e)}")
        return None

if __name__ == "__main__":
    print("Starting pose estimation prediction...")
    run_prediction()
