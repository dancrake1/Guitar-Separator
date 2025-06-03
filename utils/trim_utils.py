"""
Audio trimming utility.
"""
import os
from typing import Optional


def trim_audio(input_file: str,
               start_sec: float = 0.0,
               end_sec: Optional[float] = None,
               output_file: Optional[str] = None) -> tuple['torch.Tensor', int]:
    """
    Trim an audio file to the specified start and end times.

    Args:
        input_file: Path to the source audio file.
        start_sec: Start time in seconds.
        end_sec: End time in seconds (None to go to end).
        output_file: If provided, save trimmed audio to this path.

    Returns:
        trimmed_waveform: The trimmed audio tensor.
        sample_rate: Sample rate of the audio.
    """
    import torchaudio
    waveform, sample_rate = torchaudio.load(input_file)
    start_sample = int(start_sec * sample_rate)
    end_sample = int(end_sec * sample_rate) if end_sec is not None else waveform.shape[1]
    trimmed_waveform = waveform[:, start_sample:end_sample]
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        torchaudio.save(output_file, trimmed_waveform, sample_rate)
    return trimmed_waveform, sample_rate