import numpy as np
import torch
import torch.nn as nn


class SignalDenoiser(nn.Module):

    def __init__(self, signal_length=256):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(signal_length,512),
            nn.ReLU(),

            nn.Linear(512,512),
            nn.ReLU(),

            nn.Linear(512,signal_length)

        )

    def forward(self,x):

        return self.net(x)


class DiffusionDenoiser:

    def __init__(self,
                 signal_length=256,
                 model_path=None):

        self.signal_length=signal_length

        self.model=SignalDenoiser(signal_length)

        if model_path:

            self.model.load_state_dict(
                torch.load(
                    model_path,
                    map_location='cpu'
                )
            )

        self.model.eval()


    def preprocess(self,signal):

        signal=np.asarray(signal,dtype=np.float32)

        if len(signal)>self.signal_length:

            signal=signal[:self.signal_length]

        elif len(signal)<self.signal_length:

            pad=self.signal_length-len(signal)

            signal=np.pad(signal,(0,pad))

        return signal


    def denoise(self,signal):

        signal=self.preprocess(signal)

        x=torch.tensor(signal).float()

        with torch.no_grad():

            out=self.model(
                x.unsqueeze(0)
            )

        return out.squeeze().numpy()


denoiser=DiffusionDenoiser()


def diffusion_denoise(signal):

    return denoiser.denoise(signal)