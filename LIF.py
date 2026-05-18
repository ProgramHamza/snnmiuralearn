import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen

import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt


# --- Basic utilities and small forward-euler leaky integrator ---
def leaky_integrate_neuron(U, time_step=1e-3, I=0.0, R=5.0, C=1e-3):
  tau = R * C
  U = U + (time_step / tau) * (-U + I * R)
  return U

def plot_mem_trace(U_trace, title='Leaky Neuron', out_path=None):
  arr = np.asarray([u.item() if isinstance(u, torch.Tensor) else float(u) for u in U_trace])
  plt.figure(figsize=(8, 4))
  plt.plot(arr, marker='o')
  plt.title(title)
  plt.xlabel('Time step')
  plt.ylabel('Membrane potential U')
  plt.grid(True)
  if out_path is not None:
    plt.savefig(out_path, dpi=200)
    print(f'Saved plot to {out_path}')
  plt.show()


# --- Plotting helpers used by the tutorial ---
def plot_current_pulse_response(current, memrec, title, vline1=None, vline2=None, ylim_max=None, out_path=None):
  cur_np = current.detach().cpu().numpy().squeeze() if isinstance(current, torch.Tensor) else np.asarray(current).squeeze()
  mem_np = memrec.detach().cpu().numpy().squeeze() if isinstance(memrec, torch.Tensor) else np.asarray(memrec).squeeze()

  fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
  ax1.plot(cur_np, color='C0')
  ax1.set_ylabel('Input current')
  if vline1 is not None:
    ax1.axvline(vline1, color='k', linestyle='--')
  if vline2 is not None:
    ax1.axvline(vline2, color='k', linestyle='--')

  ax2.plot(mem_np, marker='o')
  ax2.axhline(y=1.0, color='r', linestyle='--', label='Threshold')
  ax2.set_xlabel('Time step')
  ax2.set_ylabel('Membrane potential (U)')
  if ylim_max is not None:
    ax2.set_ylim(top=ylim_max)
  ax2.legend()
  ax2.grid(True)

  fig.suptitle(title)
  fig.tight_layout()
  if out_path is not None:
    fig.savefig(out_path, dpi=200)
    print(f'Saved plot to {out_path}')
  plt.show()


def plot_cur_mem_spk(cur_in, mem_rec, spk_rec, thr_line=1.0, vline=None, ylim_max1=None, ylim_max2=None, title=None, out_path=None):
  cur_np = cur_in.detach().cpu().numpy().squeeze() if isinstance(cur_in, torch.Tensor) else np.asarray(cur_in).squeeze()
  mem_np = mem_rec.detach().cpu().numpy().squeeze() if isinstance(mem_rec, torch.Tensor) else np.asarray(mem_rec).squeeze()
  spk_np = spk_rec.detach().cpu().numpy().squeeze() if isinstance(spk_rec, torch.Tensor) else np.asarray(spk_rec).squeeze()

  fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
  ax1.plot(cur_np, color='C0')
  if vline is not None:
    ax1.axvline(vline, color='k', linestyle='--')
  ax1.set_ylabel('Input current')
  if ylim_max1 is not None:
    ax1.set_ylim(top=ylim_max1)

  ax2.plot(mem_np, label='Membrane')
  # Draw spikes as vertical markers scaled relative to membrane
  spike_scale = (mem_np.max() if mem_np.size else 1.0)
  ax2.plot(np.where(spk_np, np.arange(len(spk_np)), np.nan), spk_np * spike_scale, 'k|', label='Spikes')
  ax2.axhline(y=thr_line, color='r', linestyle='--', label='Threshold')
  ax2.set_xlabel('Time step')
  ax2.set_ylabel('Membrane / Spikes')
  if ylim_max2 is not None:
    ax2.set_ylim(top=ylim_max2)
  ax2.legend()
  ax2.grid(True)

  if title is not None:
    fig.suptitle(title)
  fig.tight_layout()
  if out_path is not None:
    fig.savefig(out_path, dpi=200)
    print(f'Saved plot to {out_path}')
  plt.show()


def plot_spk_mem_spk(spk_in, mem_rec, spk_rec, thr_line=1.0, title=None, out_path=None, ylim_max2=None):
  # spk_in: (T, N) or (T, 1) binary spike train
  spk_np = spk_in.detach().cpu().numpy().squeeze() if isinstance(spk_in, torch.Tensor) else np.asarray(spk_in).squeeze()
  mem_np = mem_rec.detach().cpu().numpy().squeeze() if isinstance(mem_rec, torch.Tensor) else np.asarray(mem_rec).squeeze()
  spk_out_np = spk_rec.detach().cpu().numpy().squeeze() if isinstance(spk_rec, torch.Tensor) else np.asarray(spk_rec).squeeze()

  T = spk_np.shape[0]
  # ensure raster shape T x N and pass as torch tensor for snntorch.spikeplot
  if isinstance(spk_in, torch.Tensor):
    raster_t = spk_in.reshape(T, -1).detach().cpu()
  else:
    raster_t = torch.from_numpy(np.asarray(spk_in).reshape(T, -1))

  fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
  splt.raster(raster_t, ax1, s=100, c='black', marker='|')
  ax1.set_ylabel('Input spikes')
  ax1.set_yticks([])

  ax2.plot(mem_np, label='Membrane')
  spike_scale = (mem_np.max() if mem_np.size else 1.0)
  ax2.plot(np.where(spk_out_np, np.arange(len(spk_out_np)), np.nan), spk_out_np * spike_scale, 'k|', label='Spikes')
  ax2.axhline(y=thr_line, color='r', linestyle='--', label='Threshold')
  ax2.set_xlabel('Time step')
  ax2.set_ylabel('Membrane / Spikes')
  if ylim_max2 is not None:
    ax2.set_ylim(top=ylim_max2)
  ax2.legend()
  ax2.grid(True)

  if title is not None:
    fig.suptitle(title)
  fig.tight_layout()
  if out_path is not None:
    fig.savefig(out_path, dpi=200)
    print(f'Saved plot to {out_path}')
  plt.show()


def compare_pulses(cur_list, mem_list, labels=None, vlines=None, title=None, out_path=None):
  # cur_list and mem_list are lists of tensors
  n = len(cur_list)
  fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
  for i in range(n):
    cur_np = cur_list[i].detach().cpu().numpy().squeeze()
    mem_np = mem_list[i].detach().cpu().numpy().squeeze()
    label = labels[i] if labels is not None else f'cur{i+1}'
    axes[0].plot(cur_np, label=label)
    axes[1].plot(mem_np, label=label)
  if vlines is not None:
    for v in vlines:
      axes[0].axvline(v, color='k', linestyle='--')
      axes[1].axvline(v, color='k', linestyle='--')
  axes[0].set_ylabel('Input current')
  axes[1].set_ylabel('Membrane potential')
  axes[1].axhline(y=1.0, color='r', linestyle='--')
  axes[1].legend()
  if title is not None:
    fig.suptitle(title)
  fig.tight_layout()
  if out_path is not None:
    fig.savefig(out_path, dpi=200)
    print(f'Saved plot to {out_path}')
  plt.show()


# --- Examples from the tutorial (runnable when script executed) ---
def run_examples():
  # basic forward-euler passive membrane
  num_steps = 100
  U = 0.9
  U_trace = []
  for step in range(num_steps):
    U_trace.append(U)
    U = leaky_integrate_neuron(U)
  plot_mem_trace(U_trace, 'Passive membrane decay (forward Euler)', out_path='leaky_decay.png')

  # Lapicque neuron
  time_step = 1e-3
  R = 5.0
  C = 1e-3
  lif1 = snn.Lapicque(R=R, C=C, time_step=time_step)

  # 3.1 Lapicque without stimulus
  num_steps = 100
  mem = torch.ones(1) * 0.9
  cur_in = torch.zeros(num_steps, 1)
  spk_out = torch.zeros(1)
  mem_rec = [mem]
  for step in range(num_steps):
    spk_out, mem = lif1(cur_in[step], mem)
    mem_rec.append(mem)
  mem_rec = torch.stack(mem_rec)
  plot_current_pulse_response(cur_in, mem_rec, "Lapicque without stimulus", out_path='lapicque_rest.png')

  # Step input example
  num_steps = 200
  cur_in = torch.cat((torch.zeros(10, 1), torch.ones(190, 1) * 0.2), 0)
  mem = torch.zeros(1)
  spk_out = torch.zeros(1)
  mem_rec = [mem]
  spk_rec = [spk_out]
  lif2 = snn.Lapicque(R=5.0, C=1e-3, time_step=time_step)
  for step in range(num_steps):
    spk_out, mem = lif2(cur_in[step], mem)
    mem_rec.append(mem)
    spk_rec.append(spk_out)
  mem_rec = torch.stack(mem_rec)
  spk_rec = torch.stack(spk_rec)
  plot_cur_mem_spk(cur_in, mem_rec, spk_rec, thr_line=1.0, vline=109, title="Lapicque step input", out_path='lapicque_step.png')

  # Pulse examples
  num_steps = 200
  cur_in1 = torch.cat((torch.zeros(10, 1), torch.ones(20, 1) * 0.1, torch.zeros(170, 1)), 0)
  cur_in2 = torch.cat((torch.zeros(10, 1), torch.ones(10, 1) * 0.111, torch.zeros(180, 1)), 0)
  cur_in3 = torch.cat((torch.zeros(10, 1), torch.ones(5, 1) * 0.147, torch.zeros(185, 1)), 0)

  def run_pulse(cur_in, filename):
    mem = torch.zeros(1)
    spk_out = torch.zeros(1)
    mem_rec = [mem]
    lif = snn.Lapicque(R=5.0, C=1e-3, time_step=time_step)
    for step in range(num_steps):
      spk_out, mem = lif(cur_in[step], mem)
      mem_rec.append(mem)
    mem_rec = torch.stack(mem_rec)
    plot_current_pulse_response(cur_in, mem_rec, f'Lapicque pulse ({filename})', vline1=10, vline2=30 if cur_in.sum()>1 else None, out_path=filename)
    return mem_rec

  mem_rec1 = run_pulse(cur_in1, 'lapicque_pulse1.png')
  mem_rec2 = run_pulse(cur_in2, 'lapicque_pulse2.png')
  mem_rec3 = run_pulse(cur_in3, 'lapicque_pulse3.png')

  compare_pulses([cur_in1, cur_in2, cur_in3], [mem_rec1, mem_rec2, mem_rec3], labels=['pulse1', 'pulse2', 'pulse3'], vlines=[10, 15, 20, 30], title="Compare pulses", out_path='compare_pulses.png')

  # Current spike input
  cur_in4 = torch.cat((torch.zeros(10, 1), torch.ones(1, 1) * 0.5, torch.zeros(189, 1)), 0)
  mem = torch.zeros(1)
  spk_out = torch.zeros(1)
  mem_rec4 = [mem]
  lif = snn.Lapicque(R=5.0, C=1e-3, time_step=time_step)
  for step in range(num_steps):
    spk_out, mem = lif(cur_in4[step], mem)
    mem_rec4.append(mem)
  mem_rec4 = torch.stack(mem_rec4)
  plot_current_pulse_response(cur_in4, mem_rec4, "Lapicque spike input", vline1=10, ylim_max=0.6, out_path='lapicque_spike.png')

  # Simple custom LIF w/ and w/o reset
  def leaky_integrate_and_fire(mem, cur=0.0, threshold=1.0, time_step=1e-3, R=5.1, C=5e-3, reset=True):
    tau_mem = R * C
    spk = (mem > threshold).float() if isinstance(mem, torch.Tensor) else float(mem > threshold)
    mem = mem + (time_step / tau_mem) * (-mem + cur * R)
    if reset:
      mem = mem - spk * threshold
    return mem, spk

  # uncontrolled spiking (no reset)
  num_steps = 200
  cur_in = torch.cat((torch.zeros(10), torch.ones(190) * 0.2), 0)
  mem = torch.zeros(1)
  mem_rec = []
  spk_rec = []
  for step in range(num_steps):
    mem, spk = leaky_integrate_and_fire(mem, cur_in[step], reset=False)
    mem_rec.append(mem)
    spk_rec.append(spk)
  mem_rec = torch.stack(mem_rec)
  spk_rec = torch.stack(spk_rec)
  plot_cur_mem_spk(cur_in, mem_rec, spk_rec, thr_line=1.0, title="Uncontrolled spiking (no reset)", out_path='uncontrolled.png')

  # with reset
  mem = torch.zeros(1)
  mem_rec = []
  spk_rec = []
  for step in range(num_steps):
    mem, spk = leaky_integrate_and_fire(mem, cur_in[step], reset=True)
    mem_rec.append(mem)
    spk_rec.append(spk)
  mem_rec = torch.stack(mem_rec)
  spk_rec = torch.stack(spk_rec)
  plot_cur_mem_spk(cur_in, mem_rec, spk_rec, thr_line=1.0, title="LIF with reset (simple)", out_path='lif_with_reset.png')

  # Lapicque neuron with periodic firing example and altered threshold
  lif3 = snn.Lapicque(R=5.1, C=5e-3, time_step=1e-3)
  cur_in = torch.cat((torch.zeros(10, 1), torch.ones(190, 1) * 0.3), 0)
  mem = torch.zeros(1)
  spk_out = torch.zeros(1)
  mem_rec = [mem]
  spk_rec = [spk_out]
  for step in range(num_steps):
    spk_out, mem = lif3(cur_in[step], mem)
    mem_rec.append(mem)
    spk_rec.append(spk_out)
  mem_rec = torch.stack(mem_rec)
  spk_rec = torch.stack(spk_rec)
  plot_cur_mem_spk(cur_in, mem_rec, spk_rec, thr_line=1.0, ylim_max2=1.3, title="Lapicque periodic firing", out_path='lapicque_periodic.png')

  # --- 3.5 Lapicque: Spike Inputs ---
  num_steps = 200
  spk_in = spikegen.rate_conv(torch.ones((num_steps,1)) * 0.40)
  print(f"There are {int(spk_in.sum())} total spikes out of {len(spk_in)} time steps.")

  fig = plt.figure(facecolor="w", figsize=(8, 1))
  ax = fig.add_subplot(111)
  splt.raster(spk_in.reshape(num_steps, -1), ax, s=100, c="black", marker="|")
  plt.title("Input Spikes")
  plt.xlabel("Time step")
  plt.yticks([])
  fig.tight_layout()
  fig.savefig('spikes_input.png', dpi=200)
  print('Saved plot to spikes_input.png')
  plt.show()

  # Initialize inputs and outputs
  mem = torch.ones(1)*0.5
  spk_out = torch.zeros(1)
  mem_rec = [mem]
  spk_rec = [spk_out]

  # Neuron simulation
  for step in range(num_steps):
    spk_out, mem = lif3(spk_in[step], mem)
    spk_rec.append(spk_out)
    mem_rec.append(mem)

  # convert lists to tensors
  mem_rec = torch.stack(mem_rec)
  spk_rec = torch.stack(spk_rec)

  plot_spk_mem_spk(spk_in, mem_rec, spk_rec, "Lapicque's Neuron Model With Input Spikes", out_path='spk_mem_spk.png')


if __name__ == '__main__':
  run_examples()
