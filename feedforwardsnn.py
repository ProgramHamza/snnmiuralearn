import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen

import torch
import torch.nn as nn
import matplotlib.pyplot as plt




def leaky_integrate_and_fire(mem, x, w, beta, threshold=1):
    spk = (mem > threshold)
    mem = beta * mem + x *w - spk*threshold
    
    return spk, mem

lif1 = snn.Leaky(beta=0.8)

#set parameters
delta_t = torch.tensor(1e-3)
tau = torch.tensor(5e-3)
beta = torch.exp(-delta_t/tau)

num_steps = 200


w=0.21
cur_in = torch.cat((torch.zeros(10), torch.ones(190)*w), 0)
mem = torch.zeros(1)
spk = torch.zeros(1)
mem_rec = []
spk_rec = []



def plot_feedforward_spikes(spk_in, spk_hidden, spk_out, title=None, out_path=None):
    """Plot input, hidden, and output spike rasters as three stacked sections."""
    layers = [
        ("Input spikes", spk_in),
        ("Hidden layer spikes", spk_hidden),
        ("Output spikes", spk_out),
    ]

    fig, axes = plt.subplots(
        len(layers),
        1,
        figsize=(10, 7),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1, 1, 0.7]},
    )

    for ax, (label, spikes) in zip(axes, layers):
        raster_data = spikes.detach().cpu().reshape(spikes.shape[0], -1)
        marker_size = 12 if raster_data.shape[1] <= 20 else 1.5
        splt.raster(raster_data, ax, s=marker_size, c="black", marker="|")
        ax.set_title(label, loc="left")
        ax.set_ylabel("Neuron")
        ax.set_xlim(0, raster_data.shape[0])

    axes[-1].set_xlabel("Time step")

    if title is not None:
        fig.suptitle(title)

    if out_path is not None:
        fig.savefig(out_path, dpi=200)
        print(f"Saved plot to {out_path}")

    plt.show()


def run_network():
    # layer parameters
    num_inputs = 784
    num_hidden = 1000
    num_outputs = 10
    beta = 0.99

    # initialize layers
    fc1 = nn.Linear(num_inputs, num_hidden)
    lif1 = snn.Leaky(beta=beta)
    fc2 = nn.Linear(num_hidden, num_outputs)
    lif2 = snn.Leaky(beta=beta)

    # Initialize hidden states
    mem1 = lif1.init_leaky()
    mem2 = lif2.init_leaky()

    # record outputs
    mem2_rec = []
    spk1_rec = []
    spk2_rec = []

    spk_in = spikegen.rate_conv(torch.rand((num_steps, num_inputs))).unsqueeze(1)

    # network simulation
    for step in range(num_steps):
        cur1 = fc1(spk_in[step]) # post-synaptic current <-- spk_in x weight
        spk1, mem1 = lif1(cur1, mem1) # mem[t+1] <--post-syn current + decayed membrane
        cur2 = fc2(spk1)
        spk2, mem2 = lif2(cur2, mem2)

        mem2_rec.append(mem2)
        spk1_rec.append(spk1)
        spk2_rec.append(spk2)

    # convert lists to tensors
    mem2_rec = torch.stack(mem2_rec)
    spk1_rec = torch.stack(spk1_rec)
    spk2_rec = torch.stack(spk2_rec)

    plot_feedforward_spikes(
        spk_in,
        spk1_rec,
        spk2_rec,
        title="Fully Connected Spiking Neural Network",
        out_path="feedforward_spikes.png",
    )


if __name__ == "__main__":
    run_network()
