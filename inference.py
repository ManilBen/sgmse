import glob
from argparse import ArgumentParser
from os.path import join

import torch
from soundfile import write
from torchaudio import load
from tqdm import tqdm

from sgmse.model import ScoreModel
from sgmse.util.other import ensure_dir, pad_spec

def inference(target_dir, noisy_file, checkpoint_file="./train_vb_29nqe0uh_epoch=115.ckpt"):
  corrector_cls = "langevin"
  ensure_dir(target_dir)
  # Settings
  sr = 16000
  snr = 0.33
  N = 50
  corrector_steps = 1
  # Load score model 
  model = ScoreModel.load_from_checkpoint(checkpoint_file, base_dir='', batch_size=16, num_workers=0, kwargs=dict(gpu=False))
  model.eval(no_ema=False)
  model.cuda()
  filename = noisy_file.split('/')[-1]
  # Load wav
  y, _ = load(noisy_file) 
  T_orig = y.size(1)   
  # Normalize
  norm_factor = y.abs().max()
  y = y / norm_factor
  # Prepare DNN input
  Y = torch.unsqueeze(model._forward_transform(model._stft(y.cuda())), 0)
  Y = pad_spec(Y)
  # Reverse sampling
  sampler = model.get_pc_sampler(
      'reverse_diffusion', corrector_cls, Y.cuda(), N=N, 
      corrector_steps=corrector_steps, snr=snr)
  sample, _ = sampler()
  # Backward transform in time domain
  x_hat = model.to_audio(sample.squeeze(), T_orig)
  # Renormalize
  x_hat = x_hat * norm_factor
  # Write enhanced wav file
  write(join(target_dir, filename), x_hat.cpu().numpy(), sr)