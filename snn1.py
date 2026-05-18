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

#dataloader
train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)


num_steps = 10
raw_vector = torch.ones(num_steps)*0.5
rate_coded_vector = torch.bernoulli(raw_vector)


data = iter(train_loader)
data_it, targets_it = next(data)

spike_data = spikegen.rate(data_it, num_steps=num_steps)

spike_data_sample = spike_data[:, 0, 0]

warnings.filterwarnings(
    'ignore',
    message='Unable to import Axes3D.*',
)




# plt.figure(facecolor="w")
# plt.subplot(1,2,1)
# plt.imshow(spike_data_sample.mean(axis=0).reshape((28,-1)).cpu(), cmap='binary')
# plt.axis('off')
# plt.title('Gain = 1')

# spike_data = spikegen.rate(data_it, num_steps=num_steps, gain=0.25)
# spike_data_sample2 = spike_data[:, 0, 0]

# plt.subplot(1,2,2)
# plt.imshow(spike_data_sample2.mean(axis=0).reshape((28,-1)).cpu(), cmap='binary')
# plt.axis('off')
# plt.title('Gain = 0.25')

# plt.show()

# # Reshape
# spike_data_sample2 = spike_data_sample2.reshape((num_steps, -1))

# # raster plot
# fig = plt.figure(facecolor="w", figsize=(10, 5))
# ax = fig.add_subplot(111)
# splt.raster(spike_data_sample2, ax, s=1.5, c="black")

# plt.title("Input Layer")
# plt.xlabel("Time step")
# plt.ylabel("Neuron Number")
# plt.show()

def convert_to_time(data, tau=5, threshold=0.01):
  spike_time = tau * torch.log(data / (data - threshold))
  print
  return spike_time


data_it_latency = torch.clamp(data_it * 0.3081 + 0.1307, 0.0, 1.0)
spike_data = spikegen.latency(data_it_latency, num_steps=100, tau=5, threshold=0.01)

fig = plt.figure(facecolor="w", figsize=(10, 5))
ax = fig.add_subplot(111)
splt.raster(spike_data[:, 0].reshape(num_steps, -1), ax, s=25, c="black")

plt.title("Input Layer")
plt.xlabel("Time step")
plt.ylabel("Neuron Number")
plt.show()

# optional save
fig.savefig('destination_path.png', format='png', dpi=300)

# Create a tensor with some fake time-series data
data = torch.Tensor([0, 1, 0, 2, 8, -20, 20, -5, 0, 1, 0])

# Plot the tensor
plt.plot(data)

plt.title("Some fake time-series data")
plt.xlabel("Time step")
plt.ylabel("Voltage (mV)")
plt.show()

spike_data = spikegen.delta(data,threshold=4,off_spike=True)

fig = plt.figure(facecolor="w", figsize=(8,1))
ax = fig.add_subplot(111)
print(spike_data)

splt.raster(spike_data, ax, c="black")

plt.title("Delta coding")
plt.xlabel("Time step")
plt.yticks([])
plt.xlim(0, len(data))
plt.show()

import HTML

spike_prob = torch.rand((num_steps, 28, 28), dtype=dtype) * 0.5
spike_rand = spikegen.rate_conv(spike_prob)
fig, ax = plt.subplots()
anim = splt.animator(spike_rand, fig, ax)

HTML(anim.to_html5_video())