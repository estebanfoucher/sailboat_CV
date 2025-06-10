import os
import shutil
from pathlib import Path
import glob
from dotenv import load_dotenv
import config

# Get model version from environment variable
MODEL_VERSION = os.getenv('MODEL_VERSION')  # Default to 'v1' if not specified

def deploy_model():
    """
    Copy the best.pt model from the latest training run to the docker YOLO model folder
    and rename it to custom-XX.pt where XX is the MODEL_VERSION
    """
    # Find the latest training directory
    runs_dir = Path("runs/detect")
    if not runs_dir.exists():
        raise FileNotFoundError(f"Training directory not found: {runs_dir}")
    
    # Get all train directories and sort them by creation time
    train_dirs = sorted(
        [d for d in runs_dir.glob("train*") if d.is_dir()],
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )
    
    if not train_dirs:
        raise FileNotFoundError("No training directories found")
    
    latest_train_dir = train_dirs[0]
    best_model_path = latest_train_dir / "weights/best.pt"
    
    if not best_model_path.exists():
        raise FileNotFoundError(f"Best model not found in {latest_train_dir}")
    
    # Define the destination path (docker YOLO model folder)
    docker_model_dir = Path("docker/yolo/models")
    docker_model_dir.mkdir(parents=True, exist_ok=True)
    
    # Define the new model name
    new_model_name = f"custom-{MODEL_VERSION}.pt"
    destination_path = docker_model_dir / new_model_name
    
    # Copy the model
    print(f"Copying model from {best_model_path} to {destination_path}")
    shutil.copy2(best_model_path, destination_path)
    print(f"Model successfully deployed as {new_model_name}")
    
if __name__ == "__main__":
    try:
        deploy_model()
    except Exception as e:
        print(f"Error deploying model: {e}") 