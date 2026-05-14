import snntorch as snn
import torch
import warnings
from pathlib import Path

from snntorch import utils
from snntorch import spikegen
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import matplotlib
import matplotlib.pyplot as plt
import snntorch.spikeplot as splt

#parameters
batch_size=128
datapath='/tmp/data/mnist'
num_classes = 10

dtype = torch.float



transform = transforms.Compose([transforms.Resize((28, 28)), 
                                transforms.Grayscale(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))])

mnist_train = datasets.MNIST(datapath, train=True, download=True, transform=transform)

#deifning subset of the data
subset=10
mnist_train = utils.data_subset(mnist_train, subset)
# ==========================================
# KEEP YOUR EXISTING CODE FOR LINES 1 to 34
# (Imports, data loading, and model.predict)
# ==========================================

print("Plot saved: active_voxel_count.png")
print("Generating 3D interactive webshow...")

import cortex
import numpy as np
import torch

# 1. Safely convert your predictions into a standard NumPy array
if torch.is_tensor(preds):
    preds_np = preds.detach().cpu().numpy()
else:
    preds_np = np.array(preds)

# 2. Extract a single frame to plot (if there are multiple time segments)
if preds_np.ndim > 1:
    vertex_data_np = preds_np.mean(axis=0) # Shows average activity across segments
else:
    vertex_data_np = preds_np

# 3. Set the correct Meta TRIBE v2 Subject Mesh
subject_name = "fsaverage5"     

try:
    # 4. Create the Pycortex Vertex object (NOT Volume!)
    # This correctly maps your 1D array of ~20k predictions directly to the cortical surface
    vertex_obj = cortex.Vertex(vertex_data_np, subject=subject_name)
    
    # 5. Launch the viewer
    print("Starting WebGL server... Open your browser to http://localhost:8080")
    cortex.webgl.show(vertex_obj, port=8080)

except Exception as e:
    print(f"\n[ERROR] Pycortex failed to draw the brain: {e}")
    print("\n[TROUBLESHOOTING]")
    print("If Pycortex says it still can't find the subject, it means 'fsaverage5' isn't in your Pycortex filestore.")
    print("Luckily, Meta included a built-in wrapper just for this. You can swap the Pycortex code above with:")
    print("from tribev2.plotting import PlotBrain")
    print("plotter = PlotBrain(mesh='fsaverage5')")
    print("# You can then feed vertex_data_np into plotter to generate the visualization.")