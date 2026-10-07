import torch
import torchvision
import transformers
import ultralytics
import sklearn
import cv2

print(f"PyTorch:        {torch.__version__}")
print(f"Torchvision:    {torchvision.__version__}")
print(f"Transformers:   {transformers.__version__}")
print(f"Ultralytics:    {ultralytics.__version__}")
print(f"Scikit-learn:   {sklearn.__version__}")
print(f"OpenCV:         {cv2.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"MPS available:  {torch.backends.mps.is_available()}")
print("\nAll dependencies OK.")
