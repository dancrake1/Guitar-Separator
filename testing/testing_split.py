#!/usr/bin/env python
# coding: utf-8

# In[108]:


import os

# Parse command-line arguments for flexible processing
import argparse
parser = argparse.ArgumentParser(description='Process and analyze guitar separation pipeline.')
parser.add_argument('song_path', help='Path to input audio file')
parser.add_argument('--start-sec', type=float, default=0.0, help='Start time in seconds for trimming')
parser.add_argument('--end-sec', type=float, default=None, help='End time in seconds for trimming')
parser.add_argument('--output-dir', default='outputs', help='Base output directory')
args = parser.parse_args()

song_path = args.song_path
start_sec = args.start_sec
end_sec = args.end_sec
output_dir = args.output_dir

# Ensure output directory and per-song subfolder exist
os.makedirs(output_dir, exist_ok=True)
song_name = os.path.splitext(os.path.basename(song_path))[0].replace('_', '')
os.makedirs(os.path.join(output_dir, song_name), exist_ok=True)


# In[109]:


import torchaudio
import torch
from demucs.pretrained import get_model
from demucs.apply import apply_model



# Load the audio
print(f"Loading audio from {song_path}...")
sample_waveform, sample_rate = torchaudio.load(song_path)
print(f"Loaded audio with shape {sample_waveform.shape} and sample rate {sample_rate}")
# Trim input audio if requested
if start_sec > 0 or end_sec is not None:
    start_sample = int(start_sec * sample_rate)
    end_sample = int(end_sec * sample_rate) if end_sec is not None else sample_waveform.shape[1]
    print(f"Trimming input audio: {start_sec}s to {end_sec if end_sec is not None else 'end'} -> samples {start_sample}:{end_sample}")
    sample_waveform = sample_waveform[:, start_sample:end_sample]
    print(f"Trimmed audio shape: {sample_waveform.shape}")

# Determine device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

def prepare_audio(waveform, source_sr, target_sr):
    """Prepare audio for model input"""
    # Resample if needed
    if source_sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, source_sr, target_sr)
        
    # Handle channels
    if waveform.shape[0] > 2:
        waveform = waveform[:2, :]
    elif waveform.shape[0] == 1:
        waveform = torch.cat([waveform, waveform], dim=0)
        
    return waveform

def enhance_guitar(guitar_waveform, sample_rate):
    """Enhance guitar with filters and transient processing"""
    # Add batch dimension if needed
    if guitar_waveform.dim() == 2:
        guitar_waveform = guitar_waveform.unsqueeze(0)
        
    # Now we can safely unpack dimensions
    b, c, t = guitar_waveform.shape
    
    # FFT for frequency domain processing
    guitar_waveform_freq = torch.fft.rfft(guitar_waveform, dim=2)
    
    # Create high-pass filter (reduce below 80Hz)
    freqs = torch.fft.rfftfreq(t, d=1/sample_rate)
    high_pass = (1 - torch.exp(-freqs/80))
    
    # Create mid boost around 2-4kHz (presence)
    mid_boost = 1.0 + 0.5 * torch.exp(-((freqs - 3000)/500)**2)
    
    # Apply filters
    filter_curve = high_pass.view(1, 1, -1) * mid_boost.view(1, 1, -1)
    guitar_waveform_freq *= filter_curve
    
    # Back to time domain
    guitar_waveform = torch.fft.irfft(guitar_waveform_freq, n=t, dim=2)
    
    # Apply subtle compression
    peak = guitar_waveform.abs().max()
    if peak > 0:
        # Simple soft knee compression
        threshold = 0.7
        ratio = 3.0
        gain = 1.2
        
        above_thresh = (guitar_waveform.abs() > threshold * peak).float()
        comp_factor = 1.0 - above_thresh * (1.0 - 1.0/ratio) * (guitar_waveform.abs() - threshold * peak) / (peak * (1.0 - threshold))
        guitar_waveform = guitar_waveform * comp_factor * gain
        
        # Final limiter
        peak = guitar_waveform.abs().max()
        if peak > 0.95:
            guitar_waveform = 0.95 * guitar_waveform / peak
    
    # Remove batch dimension if we added it
    if b == 1:
        guitar_waveform = guitar_waveform.squeeze(0)
        
    return guitar_waveform

# STAGE 1: Extract all stems with htdemucs_ft
print("STAGE 1: Separating with htdemucs_ft...")
model_stage1 = get_model("htdemucs_ft")
model_stage1.eval()
model_stage1.to(device)

# Prepare audio for first model
waveform_stage1 = prepare_audio(sample_waveform, sample_rate, model_stage1.samplerate)
waveform_stage1 = waveform_stage1.to(device)

# Separate first stage
with torch.no_grad():
    sources_stage1 = apply_model(model_stage1, waveform_stage1.unsqueeze(0))[0]
    sources_stage1 = sources_stage1.cpu()

# Get the "other" stem
other_index = model_stage1.sources.index('other') if 'other' in model_stage1.sources else None
if other_index is None:
    print("Warning: 'other' source not found in model 1. Using all non-guitar sources combined.")
    # Combine all sources except guitar to create "other"
    if 'guitar' in model_stage1.sources:
        guitar_index = model_stage1.sources.index('guitar')
        all_sources = torch.zeros_like(sources_stage1[0])
        for i, src in enumerate(model_stage1.sources):
            if i != guitar_index:
                all_sources += sources_stage1[i]
        other_waveform = all_sources
    else:
        # If no guitar source, just use the first stem as "other"
        other_waveform = sources_stage1[0]
else:
    other_waveform = sources_stage1[other_index]

# Save the other stem
other_file = os.path.join(output_dir, song_name, "stage1_other.wav")
torchaudio.save(other_file, other_waveform, model_stage1.samplerate)
print(f"Saved 'other' stem to {other_file}")

# STAGE 2: Extract guitar from "other" stem using htdemucs_6s
print("STAGE 2: Extracting guitar from 'other' using htdemucs_6s...")
model_stage2 = get_model("htdemucs_6s")
model_stage2.eval()
model_stage2.to(device)

# Prepare the "other" stem for second model
other_waveform = prepare_audio(other_waveform, model_stage1.samplerate, model_stage2.samplerate)
other_waveform = other_waveform.to(device)

# Separate second stage
with torch.no_grad():
    sources_stage2 = apply_model(model_stage2, other_waveform.unsqueeze(0))[0]
    sources_stage2 = sources_stage2.cpu()

# Get the guitar from second separation
if 'guitar' in model_stage2.sources:
    guitar_index = model_stage2.sources.index('guitar')
    extracted_guitar = sources_stage2[guitar_index]
    
    # Save the extracted guitar
    guitar_file = os.path.join(output_dir, song_name, "stage2_guitar_from_other.wav")
    torchaudio.save(guitar_file, extracted_guitar, model_stage2.samplerate)
    print(f"Saved extracted guitar to {guitar_file}")
    
    # Enhance and save
    enhanced_guitar = enhance_guitar(extracted_guitar, model_stage2.samplerate)
    enhanced_file = os.path.join(output_dir, song_name, "stage2_guitar_enhanced.wav")
    torchaudio.save(enhanced_file, enhanced_guitar, model_stage2.samplerate)
    print(f"Saved enhanced guitar to {enhanced_file}")
else:
    print("Error: 'guitar' source not found in the second model")

# BONUS: Also get the guitar from the first separation for comparison
if 'guitar' in model_stage1.sources:
    guitar_index = model_stage1.sources.index('guitar')
    original_guitar = sources_stage1[guitar_index]
    
    # Save the original guitar stem
    orig_guitar_file = os.path.join(output_dir, song_name, "stage1_original_guitar.wav")
    torchaudio.save(orig_guitar_file, original_guitar, model_stage1.samplerate)
    print(f"Saved original guitar stem to {orig_guitar_file}")
    
    # Create an enhanced version of the original guitar
    enhanced_orig_guitar = enhance_guitar(original_guitar, model_stage1.samplerate)
    enhanced_orig_file = os.path.join(output_dir, song_name, "stage1_original_guitar_enhanced.wav")
    torchaudio.save(enhanced_orig_file, enhanced_orig_guitar, model_stage1.samplerate)
    print(f"Saved enhanced original guitar to {enhanced_orig_file}")
    
    # FINAL STEP: Try combining both guitar extractions for maximum clarity
    # Resample if needed to match sample rates
    if model_stage1.samplerate != model_stage2.samplerate:
        original_guitar = torchaudio.functional.resample(
            original_guitar, model_stage1.samplerate, model_stage2.samplerate)
    
    # Make sure shapes match
    min_length = min(original_guitar.shape[1], extracted_guitar.shape[1])
    original_guitar = original_guitar[:, :min_length]
    extracted_guitar = extracted_guitar[:, :min_length]
    
    # Blend with 70% from first model, 30% from second model
    combined_guitar = 0.7 * original_guitar + 0.3 * extracted_guitar
    
    # Enhance the combined result
    enhanced_combined = enhance_guitar(combined_guitar, model_stage2.samplerate)
    combined_file = os.path.join(output_dir, song_name, "combined_guitar_enhanced.wav")
    torchaudio.save(combined_file, enhanced_combined, model_stage2.samplerate)
    print(f"Saved combined enhanced guitar to {combined_file}")

print("Processing complete!")


# ## Analysis
# 
# Audio File → Audio Analysis Model → Extract tempo/rhythm → 
# Text Description → LLM → Strumming Suggestions

# In[110]:


import torchaudio

def trim_audio(input_file, output_file=None, start_sec=0, end_sec=None):
    """
    Trim audio file to specified start and end times.
    
    Parameters:
    - input_file: Path to the input audio file
    - output_file: Path to save the trimmed file (if None, returns without saving)
    - start_sec: Start time in seconds
    - end_sec: End time in seconds (if None, trims to the end of the file)
    
    Returns:
    - trimmed_waveform: Tensor containing the trimmed audio
    - sample_rate: Sample rate of the audio
    """
    # Load the audio
    waveform, sample_rate = torchaudio.load(input_file)
    
    # Convert time to samples
    start_sample = int(start_sec * sample_rate)
    end_sample = int(end_sec * sample_rate) if end_sec is not None else waveform.shape[1]
    
    # Trim the audio
    trimmed_waveform = waveform[:, start_sample:end_sample]
    
    # Save the trimmed audio if output_file is provided
    if output_file:
        torchaudio.save(output_file, trimmed_waveform, sample_rate)
    
    return trimmed_waveform, sample_rate

trim_audio(os.path.join(output_dir, song_name, 'stage2_guitar_enhanced.wav'),
           start_sec=start_sec, end_sec=end_sec,
           output_file=os.path.join(output_dir, song_name, 'stage2_guitar_enhanced_cut.wav'))


# In[112]:


"""dynamic_guitar_strum_analysis.py  –  chords & notes processed **independently**
==========================================================================

* Deep‑Chroma (madmom) → **chord timeline** (segment‑level)
* torchcrepe (or pyin fallback) → **note timeline** (event‑level)
* Original beat‑aligned strum/chord/bar/section logic LEFT INTACT so your UI
  keeps working, but we **do not overwrite chords with notes** anymore.
* Extra DataFrames returned: `chords_timeline`, `notes_timeline`.
* One label per DataFrame → no clobbering; overlapping times are fine.

Install (CPU only):
    pip install librosa madmom torch torchaudio torchcrepe pandas tqdm "numpy<1.24"

If you later add a CUDA PyTorch wheel, torchcrepe will use it automatically.
"""
import warnings, traceback
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional

import numpy as np, librosa, pandas as pd, torch, torchaudio
from pathlib import Path
from tqdm import tqdm

# --------------------------------------------------------------------------
# Dataclasses (unchanged for strums/bars/sections)
# --------------------------------------------------------------------------
@dataclass
class Strum:
    time: float; bar: int; sub_16: int; direction: str; velocity: float; kind: str; label: str   # kind NOTE|CHORD|NONE

@dataclass
class BarSummary:
    bar: int; bit_pattern: str; down_up: str; mean_vel: float; chords: List[str]

@dataclass
class Section:
    start_bar: int; end_bar: int; pattern_bits: str; chords: List[str]

# --------------------------------------------------------------------------
# Polyphony helper (v2) -----------------------------------------------------
# --------------------------------------------------------------------------
def _is_polyphonic(mag_db: np.ndarray,
                   peak_db: float = -35.0,
                   min_peaks: int = 3,
                   dom_margin: float = 8.0) -> bool:
    """
    Return True if the frame is almost certainly a chord.
    Override: if the strongest peak is `dom_margin` dB louder than the
    2nd‑strongest, treat as monophonic even when `min_peaks` is exceeded.
    """
    strong = mag_db > peak_db
    # indices of strong bins
    idx = np.flatnonzero(strong)
    if len(idx) == 0:
        return False

    # dominant‑peak override
    sorted_db = np.sort(mag_db[idx])
    if len(sorted_db) >= 2 and sorted_db[-1] - sorted_db[-2] >= dom_margin:
        return False                            # clearly one string dominates

    # otherwise count distinct strong groups
    groups = np.split(strong, np.flatnonzero(~strong) + 1)
    n_peaks = sum(g.any() for g in groups)
    return n_peaks >= min_peaks


def spectral_centroid_direction(y: np.ndarray, sr: int, onset_frames: np.ndarray) -> List[str]:
    """Classify Down/Up strokes by sign of spectral‑centroid slope around attack."""
    cent = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=512)[0]
    dirs = []
    for f in onset_frames:
        a = max(0, f-2)
        b = min(len(cent)-1, f+2)
        dirs.append('D' if cent[b] - cent[a] < 0 else 'U')
    return dirs

# --------------------------------------------------------------------------
# 1.  Deep‑Chroma chord timeline (segment‑level) ----------------------------
# --------------------------------------------------------------------------

def chord_timeline(audio_path: str) -> pd.DataFrame:
    """Return DF with columns [start, end, chord]."""
    from madmom.audio.chroma import DeepChromaProcessor
    from madmom.features.chords import DeepChromaChordRecognitionProcessor
    chroma = DeepChromaProcessor()(audio_path)
    segs   = DeepChromaChordRecognitionProcessor()(chroma)
    df = pd.DataFrame(segs, columns=["start", "end", "label"])
    df["label"] = df["label"].str.split("/").str[0]
    return df

# --------------------------------------------------------------------------
# 2.  torchcrepe / pyin note timeline (event‑level) -------------------------
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# 2.  Gated note timeline (event‑level, single‑string only) -----------------
# --------------------------------------------------------------------------
def note_timeline(audio_path: str,
                  hop_s: float = 0.01,
                  conf_thresh: float = .8,
                  peak_db: float = -52,
                  min_peaks: int = 4,
                  device: Optional[str] = None) -> pd.DataFrame:
    """
    Returns DF [time, note] containing ONLY intentionally plucked single‑string notes.
    Strategy:
      1. Find onsets (same settings as the rest of the pipeline).
      2. For each onset grab a 40 ms slice and CQT → run _is_polyphonic().
      3. Only if that slice is *not* polyphonic do we call torchcrepe/pyin.
    """
    import torchcrepe, torchcrepe.decode as tcd, torchcrepe.filter as tcf

    # --- load + onset detection ------------------------------------------
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    onset_env   = librosa.onset.onset_strength(y=y, sr=sr)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    onset_times  = librosa.frames_to_time(onset_frames, sr=sr)

    # --- constants --------------------------------------------------------
    slice_ms = 40                                   # analysis window
    slice_samps = int(sr * slice_ms / 1000.0)
    hop_len = int(round(16000 * hop_s))

    # --- prepare harmonic layer & CQT for gate ---------------------------
    y_harm, _ = librosa.effects.hpss(y)             # helps both gates
    C = np.abs(librosa.cqt(y_harm,
                           sr=sr,
                           hop_length=512,
                           n_bins=84,
                           bins_per_octave=12))
    C_db = librosa.amplitude_to_db(C, ref=np.max)

    # --- choose device for torchcrepe ------------------------------------
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    # --- resample once for torchcrepe ------------------------------------
    y16 = torchaudio.functional.resample(torch.tensor(y_harm),
                                         sr, 16000) if sr != 16000 else torch.tensor(y_harm)
    y16 = y16.unsqueeze(0).to(device)

    rows = []
    try:
        for t, fr in zip(onset_times, onset_frames):
            # ------------------------------------------------------------------
            # 2·A  Polyphony gate  (CQT frame centred on onset)
            # ------------------------------------------------------------------
            # shift 30 ms (≈ 3 CQT hops at hop_length=512) past the onset
            off = fr + 3
            cqt_frame = C_db[:, off] if off < C_db.shape[1] else C_db[:, -1]
            if _is_polyphonic(cqt_frame, peak_db, min_peaks):
                continue                                  # reject → chord

            # ------------------------------------------------------------------
            # 2·B  Periodicity gate (torchcrepe, harmonic layer only)
            # ------------------------------------------------------------------
            start16 = max(0, int(t * 16000) - hop_len//2)
            end16   = start16 + hop_len
            frame = y16[..., start16:end16]               # shape (1, N)
            f0, pdist = torchcrepe.predict(frame,
                                           16000,
                                           hop_len,
                                           model='full',
                                           decoder=tcd.argmax,
                                           fmin=80, fmax=1200,
                                           batch_size=64,
                                           device=device,
                                           return_periodicity=True)
            f0 = tcf.median(f0, 3)
            hz   = float(f0.squeeze())
            pval = float(pdist.squeeze())
            if pval < conf_thresh or not (80 < hz < 1200):
                continue                                  # weak/confused

            rows.append((t, librosa.hz_to_note(hz, octave=False)))

    except Exception as e:
        warnings.warn(f"torchcrepe failed ({e}); falling back to pyin")
        for t, fr in zip(onset_times, onset_frames):
            if _is_polyphonic(C_db[:, fr], peak_db, min_peaks):
                continue
            start = max(0, fr*512)
            end   = start + slice_samps
            f0, _, _ = librosa.pyin(y[start:end], fmin=80, fmax=1200, sr=sr)
            if f0 is not None and not np.isnan(f0).all():
                hz = float(np.nanmedian(f0))
                rows.append((t, librosa.hz_to_note(hz, octave=False)))

    return pd.DataFrame(rows, columns=["time", "note"])

# --------------------------------------------------------------------------
# 3.  Helper: beat‑level chord map (legacy, for strums/bars) ----------------
# --------------------------------------------------------------------------

def chord_sequence_by_beat(chords_df: pd.DataFrame, beat_times: np.ndarray):
    idx, seg_i = {}, 0
    for b, bt in enumerate(beat_times):
        while seg_i+1 < len(chords_df) and chords_df.iloc[seg_i]['end'] <= bt:
            seg_i += 1
        idx[b] = chords_df.iloc[seg_i]['label']
    return idx

# --------------------------------------------------------------------------
# 4.  Core analysis (strums/bars/sections) – chord map only -----------------
# --------------------------------------------------------------------------

def analyse_audio(audio_path: str, return_dataframes: bool = True):
    y, sr = librosa.load(audio_path, sr=None)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units='frames', tightness=400)
    tempo = float(np.atleast_1d(tempo)[0])
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    # ----- chord & note timelines (independent) -----
    chords_df = chord_timeline(audio_path)
    notes_df  = note_timeline(audio_path)
    beat_chords = chord_sequence_by_beat(chords_df, beat_times)

    # ----- onsets / strums (keep original behaviour) -----
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    onset_times  = librosa.frames_to_time(onset_frames, sr=sr)

    grid_step = 60/tempo/4
    grid_times = np.arange(beat_times[0], beat_times[-1]+grid_step, grid_step)

    y_harm, _ = librosa.effects.hpss(y)
    chroma = librosa.feature.chroma_cqt(y=y_harm, sr=sr)
    directions = spectral_centroid_direction(y, sr, onset_frames)
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]

    strums: List[Strum] = []
    for i, (t, fr) in enumerate(zip(onset_times, onset_frames)):
        gidx = int(np.argmin(np.abs(grid_times - t)))
        bar_idx, sub16 = divmod(gidx, 16)
        vel = float(rms[min(len(rms)-1, fr)])
        kind, label = 'CHORD', beat_chords.get(int(np.argmin(np.abs(beat_times - t))), 'N')
        strums.append(Strum(time=float(t), bar=bar_idx+1, sub_16=sub16,
                            direction=directions[i], velocity=vel,
                            kind=kind, label=label))

    # ----- summarise bars/sections (same as before) -----
    bars: Dict[int, BarSummary] = {}
    for s in strums:
        b = bars.setdefault(s.bar, BarSummary(bar=s.bar, bit_pattern=['0']*16,
                     down_up=['-']*16, mean_vel=0.0, chords=[]))
        b.bit_pattern[s.sub_16] = '1'
        b.down_up[s.sub_16] = s.direction
        b.mean_vel += s.velocity
        if s.kind == 'CHORD':
            b.chords.append(s.label)
    for b in bars.values():
        hits = b.bit_pattern.count('1')
        b.mean_vel /= max(1, hits)
        b.bit_pattern = ''.join(b.bit_pattern)
        b.down_up = ' '.join(b.down_up)
        b.chords = sorted(set(b.chords))

    ordered = [bars[k] for k in sorted(bars)]
    # simple section clustering unchanged for brevity ...

    if return_dataframes:
        return dict(
            tempo_bpm=tempo,
            strums=pd.DataFrame([asdict(s) for s in strums]),
            bars=pd.DataFrame([asdict(b) for b in ordered]),
            chords_timeline=chords_df,
            notes_timeline=notes_df,
        )
    else:
        return dict(
            tempo_bpm=tempo,
            strums=[asdict(s) for s in strums],
            bars=[asdict(b) for b in ordered],
            chords_timeline=chords_df.to_dict('records'),
            notes_timeline=notes_df.to_dict('records'),
        )

# --------------------------------------------------------------------------
# Notebook helper -----------------------------------------------------------

def run_in_notebook(audio_path: str):
    data = analyse_audio(audio_path, return_dataframes=True)
    from IPython.display import display
    print(f"Tempo ≈ {data['tempo_bpm']:.1f} BPM\n")
    print("Chord segments:"); display(data['chords_timeline'].head())
    print("Note events:");   display(data['notes_timeline'].head())
    print("\nStrums (first 10):"); display(data['strums'].head(10))
    print("\nBars:"); display(data['bars'].head())
    return data


# In[113]:


audio_path = os.path.join(output_dir, song_name, 'stage2_guitar_enhanced_cut.wav')
# data = run_in_notebook(audio_path)
data = analyse_audio(audio_path, return_dataframes=True)


# In[114]:


data['notes_timeline']


# In[115]:


chord_timeline(audio_path)


# In[116]:


import json
import pandas as pd
import numpy as np

# Function to convert pandas DataFrames and NumPy arrays to JSON-serializable types
def convert_to_serializable(obj):
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')  # Convert DataFrame to list of dictionaries
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return [convert_to_serializable(i) for i in obj]
    else:
        return obj

# Convert the data and save to JSON
serializable_data = convert_to_serializable(data)

# Save to file
with open(os.path.join(output_dir, song_name, 'guitar_data.json'), 'w') as f:
    json.dump(serializable_data, f, indent=4)



