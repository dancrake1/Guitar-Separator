"""
Stage 1 separation: extract the 'other' stem using htdemucs_ft.
"""
import os
import torch
import torchaudio
from demucs.pretrained import get_model
from demucs.apply import apply_model


def prepare_audio(waveform: torch.Tensor, source_sr: int, target_sr: int) -> torch.Tensor:
    # Resample if needed
    if source_sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, source_sr, target_sr)
    # Ensure 2 channels
    if waveform.shape[0] > 2:
        waveform = waveform[:2, :]
    elif waveform.shape[0] == 1:
        waveform = torch.cat([waveform, waveform], dim=0)
    return waveform


def run_stage1(waveform: torch.Tensor, sample_rate: int, device: torch.device):
    """
    Run the first Demucs stage ('htdemucs_ft') to extract the 'other' stem.
    Returns the waveform of the 'other' stem and the model sample rate.
    """
    model = get_model("htdemucs_ft")
    model.eval()
    model.to(device)
    proc_wave = prepare_audio(waveform, sample_rate, model.samplerate)
    proc_wave = proc_wave.to(device)
    with torch.no_grad():
        sources = apply_model(model, proc_wave.unsqueeze(0))[0].cpu()
    if 'other' in model.sources:
        index = model.sources.index('other')
        other = sources[index]
    else:
        # Combine all non-guitar stems
        if 'guitar' in model.sources:
            gi = model.sources.index('guitar')
            other = sum(s for i, s in enumerate(sources) if i != gi)
        else:
            other = sources[0]
    return other, model.samplerate


def save_stage1(other_waveform: torch.Tensor,
                sr: int,
                base_dir: str,
                song_name: str) -> str:
    """
    Save the 'other' stem waveform to disk under base_dir/song_name.
    """
    subdir = os.path.join(base_dir, song_name)
    os.makedirs(subdir, exist_ok=True)
    out_path = os.path.join(subdir, 'stage1_other.wav')
    torchaudio.save(out_path, other_waveform, sr)
    return out_path