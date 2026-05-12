import snntorch as snn
import torch
import warnings
from pathlib import Path

from snntorch import utils
from snntorch import spikegen
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


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

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import snntorch.spikeplot as splt
fig, ax = plt.subplots()
anim = splt.animator(spike_data_sample, fig, ax)
# plt.rcParams['animation.ffmpeg_path'] = 'C:\\path\\to\\your\\ffmpeg.exe'

output_path = Path(__file__).with_name('mnist_animation.html')
output_path.write_text(anim.to_jshtml(), encoding='utf-8')
print(f'Saved animation to {output_path}')