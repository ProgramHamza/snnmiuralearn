import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen
import cv2

from torch import device
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import numpy as np
import itertools
from model import Net


from video_processor import VideoProcessor

lif1 = snn.Leaky(beta=0.9)

# dataloader arguments
batch_size = 128
dir_path='/srv/tribe-share/enma/saved_mri'

num_steps = 25
beta = 0.95

dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

video_train = VideoProcessor(dir_path).iterate_train_videos()
video_test = VideoProcessor(dir_path).iterate_test_videos()

# Create DataLoaders
train_loader = DataLoader(video_train, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(video_test, batch_size=batch_size, shuffle=True, drop_last=True)

net = Net().to(device)
def ccc_loss(pred, target):
    pred_m    = pred.mean()
    target_m  = target.mean()
    covar     = ((pred - pred_m) * (target - target_m)).mean()
    pred_v    = pred.var()
    target_v  = target.var()
    ccc       = (2 * covar) / (pred_v + target_v + (pred_m - target_m)**2 + 1e-8)
    return 1 - ccc  # 0 = perfect, 2 = worst

def total_loss(pred, target):
    # pred/target shape: [B, 2]  (valence, arousal)
    mse   = nn.MSELoss()(pred, target)
    ccc_v = ccc_loss(pred[:, 0], target[:, 0])  # valence
    ccc_a = ccc_loss(pred[:, 1], target[:, 1])  # arousal
    return mse + 0.5 * (ccc_v + ccc_a)
optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, betas=(0.9, 0.999))

data, targets = next(iter(train_loader))
data = data.to(device)
targets = targets.to(device)


spk_rec, mem_rec = net(data.view(batch_size, -1))

# initialize the total loss value
loss_val = torch.zeros((1), dtype=dtype, device=device)

# sum loss at every step
for step in range(num_steps):
  loss_val += total_loss(mem_rec[step], targets)

# pass data into the network, sum the spikes over time
# and compare the neuron with the highest number of spikes
# with the target

def print_batch_accuracy(data, targets, train=False):
    output, _ = net(data.view(batch_size, -1))
    _, idx = output.sum(dim=0).max(1)
    acc = np.mean((targets == idx).detach().cpu().numpy())

    if train:
        print(f"Train set accuracy for a single minibatch: {acc*100:.2f}%")
    else:
        print(f"Test set accuracy for a single minibatch: {acc*100:.2f}%")

def train_printer():
    print(f"Epoch {epoch}, Iteration {iter_counter}")
    print(f"Train Set Loss: {loss_hist[counter]:.2f}")
    print(f"Test Set Loss: {test_loss_hist[counter]:.2f}")
    print_batch_accuracy(data, targets, train=True)
    print_batch_accuracy(test_data, test_targets, train=False)
    print("\n")


num_epochs = 100
loss_hist = []
test_loss_hist = []
counter = 0

# Outer training loop
for epoch in range(num_epochs):
    iter_counter = 0
    train_batch = iter(train_loader)

    # Minibatch training loop
    for data, targets in train_batch:
        data = data.to(device)
        targets = targets.to(device)

        # forward pass
        net.train()
        spk_rec, mem_rec = net(data.view(batch_size, -1))

        # initialize the loss & sum over time
        loss_val = torch.zeros((1), dtype=dtype, device=device)
        for step in range(num_steps):
            loss_val += total_loss(mem_rec[step], targets)

        # Gradient calculation + weight update
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        # Store loss history for future plotting
        loss_hist.append(loss_val.item())

        # Test set
        with torch.no_grad():
            net.eval()
            test_data, test_targets = next(iter(test_loader))
            test_data = test_data.to(device)
            test_targets = test_targets.to(device)

            # Test set forward pass
            test_spk, test_mem = net(test_data.view(batch_size, -1))

            # Test set loss
            test_loss = torch.zeros((1), dtype=dtype, device=device)
            for step in range(num_steps):
                test_loss += total_loss(test_mem[step], test_targets)
            test_loss_hist.append(test_loss.item())

            # Print train/test loss/accuracy
            if counter % 50 == 0:
                train_printer()
            counter += 1
            iter_counter +=1

# Plot Loss
fig = plt.figure(facecolor="w", figsize=(10, 5))
plt.plot(loss_hist)
plt.plot(test_loss_hist)
plt.title("Loss Curves")
plt.legend(["Train Loss", "Test Loss"])
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()