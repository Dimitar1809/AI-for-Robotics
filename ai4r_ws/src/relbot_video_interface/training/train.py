from ultralytics import YOLO
import os
from pathlib import Path

def train_yolo():
    # Get the package directory
    pkg_dir = Path(__file__).parent.parent
    
    # Load a model
    model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
    
    # Train the model
    results = model.train(
        data=str(pkg_dir / 'training' / 'data.yaml'),
        epochs=100,
        imgsz=640,
        batch=16,
        name='yolov8n_person_helmet',
        patience=20,  # early stopping patience
        save=True,  # save best checkpoint
        device='0' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu'  # use GPU if available
    )
    
    # Save the trained model
    model.export(format='onnx')  # export to ONNX format
    model.export(format='torchscript')  # export to TorchScript format

if __name__ == '__main__':
    train_yolo() 