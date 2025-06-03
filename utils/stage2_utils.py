"""
Stage 2 separation: extract and enhance guitar from the 'other' stem using htdemucs_6s.
"""
import os
import torch
import torchaudio
from demucs.pretrained import get_model
from demucs.apply import apply_model
from .stage1_utils import prepare_audio


def enhance_guitar(guitar_waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Enhance guitar with filters and simple compression."""
    # Ensure batch dimension
    gw = guitar_waveform
    if gw.dim() == 2:
        gw = gw.unsqueeze(0)
    b, c, t = gw.shape
    # FFT domain processing
    freq = torch.fft.rfft(gw, dim=2)
    freqs = torch.fft.rfftfreq(t, d=1/sample_rate)
    # High-pass below ~80Hz
    high_pass = (1 - torch.exp(-freqs/80)).view(1, 1, -1)
    # Mid boost around 3kHz
    mid_boost = (1.0 + 0.5 * torch.exp(-((freqs - 3000)/500)**2)).view(1, 1, -1)
    filter_curve = high_pass * mid_boost
    freq = freq * filter_curve
    gw = torch.fft.irfft(freq, n=t, dim=2)
    # Simple compression
    peak = gw.abs().max()
    if peak > 0:
        threshold = 0.7
        ratio = 3.0
        gain = 1.2
        above = (gw.abs() > threshold * peak).float()
        comp = 1.0 - above * (1.0 - 1.0/ratio) * (gw.abs() - threshold * peak) / (peak * (1.0 - threshold))
        gw = gw * comp * gain
        peak = gw.abs().max()
        if peak > 0.95:
            gw = 0.95 * gw / peak
    # Remove batch if added
    if b == 1:
        gw = gw.squeeze(0)
    return gw


def run_stage2(other_waveform: torch.Tensor,
               other_sr: int,
               device: torch.device) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Run second Demucs stage ('htdemucs_6s') to extract and enhance the guitar.
    Returns the raw and enhanced guitar waveforms and the sample rate.
    """
    model = get_model("htdemucs_6s")
    model.eval()
    model.to(device)
    proc_wave = prepare_audio(other_waveform, other_sr, model.samplerate)
    proc_wave = proc_wave.to(device)
    with torch.no_grad():
        sources = apply_model(model, proc_wave.unsqueeze(0))[0].cpu()
    if 'guitar' in model.sources:
        idx = model.sources.index('guitar')
        guitar = sources[idx]
        enhanced = enhance_guitar(guitar, model.samplerate)
    else:
        raise RuntimeError("'guitar' source not found in second model")
    return guitar, enhanced, model.samplerate


def save_stage2(guitar_waveform: torch.Tensor,
                enhanced_waveform: torch.Tensor,
                sr: int,
                base_dir: str,
                song_name: str) -> tuple[str, str]:
    """
    Save the extracted and enhanced guitar to disk.
    Returns tuple of (raw_path, enhanced_path).
    """
    subdir = os.path.join(base_dir, song_name)
    os.makedirs(subdir, exist_ok=True)
    raw_path = os.path.join(subdir, 'stage2_guitar_from_other.wav')
    torchaudio.save(raw_path, guitar_waveform, sr)
    enhanced_path = os.path.join(subdir, 'stage2_guitar_enhanced.wav')
    torchaudio.save(enhanced_path, enhanced_waveform, sr)
    return raw_path, enhanced_path