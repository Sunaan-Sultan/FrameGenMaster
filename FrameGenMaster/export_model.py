import torch
import torch.nn as nn
import onnx

print("--- AI Model Generation Phase ---")

# 1. Defining a Very Basic CNN for Frame Interpolation
class BasicFrameInterpolator(nn.Module):
    def __init__(self):
        super(BasicFrameInterpolator, self).__init__()
        # Input: 6 channels (3 channels for Frame A + 3 channels for Frame B)
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        # Output: 3 channels (The generated intermediate frame)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, padding=1)

    def forward(self, frame_a, frame_b):
        # Concatenate Frame A and Frame B along the channel dimension
        x = torch.cat((frame_a, frame_b), dim=1) 
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x

# Initialize the model
model = BasicFrameInterpolator()
# Set model to evaluation mode (crucial for exporting)
model.eval() 
print("[SUCCESS] PyTorch Model initialized.")

# 2. Creating Dummy Input Data (to trace the model's architecture)
# Shape: (Batch Size, Channels, Height, Width)
# We are using 1080p resolution (1920x1080) with 3 color channels (RGB)
dummy_frame_a = torch.randn(1, 3, 1080, 1920)
dummy_frame_b = torch.randn(1, 3, 1080, 1920)
print("[SUCCESS] Dummy 1080p frames created.")

# 3. Exporting the model to ONNX format
onnx_file_path = "frame_interpolator.onnx"

torch.onnx.export(
    model, 
    (dummy_frame_a, dummy_frame_b),           # Model inputs
    onnx_file_path,                           # Where to save the file
    export_params=True,                       # Store the trained parameter weights inside the model file
    opset_version=11,                         # ONNX version
    do_constant_folding=True,                 # Optimize constant folding for inference
    input_names=['frame_a', 'frame_b'],       # Set names for the inputs
    output_names=['generated_frame'],         # Set name for the output
    dynamic_axes={                            # Enable dynamic batch sizing (optional, but good practice)
        'frame_a': {0: 'batch_size'},    
        'frame_b': {0: 'batch_size'},
        'generated_frame': {0: 'batch_size'}
    }
)

print(f"\n[SUCCESS] Model successfully exported to: {onnx_file_path}")