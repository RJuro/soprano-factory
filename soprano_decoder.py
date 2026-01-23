"""
Soprano Decoder - Vocos-based audio decoder for Soprano-80M.

This decoder converts LLM hidden states to audio waveforms.
Architecture matches ekwek/Soprano-80M (512-dim version).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ConvNeXtBlock(nn.Module):
    """ConvNeXt block for 1D audio processing."""

    def __init__(self, dim, intermediate_dim, kernel_size=3):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = nn.Parameter(torch.ones(dim) * 1e-6)

    def forward(self, x):
        # x: (B, C, T)
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # (B, T, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = F.gelu(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        x = x.transpose(1, 2)  # (B, C, T)
        return residual + x


class ISTFTHead(nn.Module):
    """ISTFT-based head that converts features to audio."""

    def __init__(self, dim, n_fft=4096, hop_length=512):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.out = nn.Linear(dim, n_fft + 2)  # magnitude + phase

    def forward(self, x):
        # x: (B, C, T) -> (B, T, C)
        x = x.transpose(1, 2)
        x = self.out(x)  # (B, T, n_fft + 2)

        # Split into magnitude and phase
        mag = x[..., :self.n_fft // 2 + 1]
        phase = x[..., self.n_fft // 2 + 1:]

        # Ensure positive magnitude
        mag = F.softplus(mag)

        # Build complex spectrogram
        real = mag * torch.cos(phase)
        imag = mag * torch.sin(phase)
        spec = torch.complex(real, imag)  # (B, T, n_fft // 2 + 1)
        spec = spec.transpose(1, 2)  # (B, n_fft // 2 + 1, T)

        # ISTFT
        window = torch.hann_window(self.n_fft, device=x.device)
        audio = torch.istft(spec, self.n_fft, hop_length=self.hop_length, window=window)

        return audio


class SopranoDecoder(nn.Module):
    """
    Vocos-based decoder for Soprano-80M.

    Takes LLM hidden states and produces audio waveforms.
    Architecture: Conv embed -> 8x ConvNeXt blocks -> ISTFT head
    """

    def __init__(self, dim=512, intermediate_dim=1536, num_layers=8,
                 n_fft=4096, hop_length=512, sample_rate=32000):
        super().__init__()
        self.dim = dim
        self.sample_rate = sample_rate

        # Input projection
        self.embed = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(dim)

        # ConvNeXt blocks
        self.convnext = nn.ModuleList([
            ConvNeXtBlock(dim, intermediate_dim) for _ in range(num_layers)
        ])

        # Output
        self.final_layer_norm = nn.LayerNorm(dim)
        self.head = ISTFTHead(dim, n_fft=n_fft, hop_length=hop_length)

    def forward(self, hidden_states):
        """
        Args:
            hidden_states: (B, T, dim) - LLM hidden states for audio tokens

        Returns:
            audio: (B, samples) - Audio waveform at 32kHz
        """
        # (B, T, C) -> (B, C, T)
        x = hidden_states.transpose(1, 2)

        # Embed
        x = self.embed(x)
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)

        # ConvNeXt blocks
        for block in self.convnext:
            x = block(x)

        # Final norm and head
        x = self.final_layer_norm(x.transpose(1, 2)).transpose(1, 2)
        audio = self.head(x)

        return audio


def load_decoder(checkpoint_path, device='cuda'):
    """Load decoder from checkpoint with architecture auto-detection."""
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Detect architecture from checkpoint
    # Check embed weight shape: (out_channels, in_channels, kernel_size)
    embed_weight = state_dict.get('decoder.embed.weight', state_dict.get('embed.weight'))
    if embed_weight is not None:
        dim = embed_weight.shape[0]
    else:
        dim = 512  # Default for Soprano-80M

    # Count convnext layers
    num_layers = sum(1 for k in state_dict.keys() if 'convnext' in k and 'gamma' in k)

    print(f"Detected decoder: dim={dim}, num_layers={num_layers}")

    # Build decoder
    decoder = SopranoDecoder(dim=dim, intermediate_dim=dim * 3, num_layers=num_layers)

    # Handle potential 'decoder.' prefix in keys
    if any(k.startswith('decoder.') for k in state_dict.keys()):
        # Strip 'decoder.' prefix
        state_dict = {k.replace('decoder.', ''): v for k, v in state_dict.items()
                      if k.startswith('decoder.')}

    # Handle 'head.' vs 'head.out.' naming
    if 'head.weight' in state_dict:
        state_dict['head.out.weight'] = state_dict.pop('head.weight')
        state_dict['head.out.bias'] = state_dict.pop('head.bias')

    decoder.load_state_dict(state_dict, strict=False)
    decoder.to(device).eval()

    return decoder
