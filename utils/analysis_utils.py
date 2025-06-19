"""
Advanced guitar analysis utilities with chord and note detection.
Modularized from the original dynamic_guitar_strum_analysis.py logic.
"""
import warnings
import traceback
import json
import numpy as np
import pandas as pd
import librosa
import torch
import torchaudio
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from tqdm import tqdm


# --------------------------------------------------------------------------
# Dataclasses
# --------------------------------------------------------------------------
@dataclass
class Strum:
    time: float
    bar: int
    sub_16: int
    direction: str
    velocity: float
    kind: str
    label: str  # kind NOTE|CHORD|NONE


@dataclass
class BarSummary:
    bar: int
    bit_pattern: str
    down_up: str
    mean_vel: float
    chords: List[str]


@dataclass
class Section:
    start_bar: int
    end_bar: int
    pattern_bits: str
    chords: List[str]


# --------------------------------------------------------------------------
# Helper Functions
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
        return False  # clearly one string dominates

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
# Chord Timeline (Deep-Chroma)
# --------------------------------------------------------------------------
def chord_timeline(audio_path: str) -> pd.DataFrame:
    """Return DF with columns [start, end, chord]."""
    try:
        from madmom.audio.chroma import DeepChromaProcessor
        from madmom.features.chords import DeepChromaChordRecognitionProcessor
        
        chroma = DeepChromaProcessor()(audio_path)
        segs = DeepChromaChordRecognitionProcessor()(chroma)
        df = pd.DataFrame(segs, columns=["start", "end", "label"])
        df["label"] = df["label"].str.split("/").str[0]
        return df
    except ImportError:
        warnings.warn("madmom not available, using basic chord detection")
        # Fallback to basic chord detection
        y, sr = librosa.load(audio_path, sr=None)
        duration = len(y) / sr
        # Create simple chord segments every 2 seconds
        times = np.arange(0, duration, 2.0)
        chords = ['G'] * len(times)  # Placeholder
        df_data = []
        for i, start in enumerate(times):
            end = min(start + 2.0, duration)
            df_data.append({'start': start, 'end': end, 'label': chords[i]})
        return pd.DataFrame(df_data)


# --------------------------------------------------------------------------
# Note Timeline (TorchCrepe)
# --------------------------------------------------------------------------
def note_timeline(audio_path: str,
                  hop_s: float = 0.01,
                  conf_thresh: float = 0.8,
                  peak_db: float = -52,
                  min_peaks: int = 4,
                  device: Optional[str] = None) -> pd.DataFrame:
    """
    Returns DF [time, note] containing ONLY intentionally plucked single‑string notes.
    """
    try:
        import torchcrepe
        import torchcrepe.decode as tcd
        import torchcrepe.filter as tcf
    except ImportError:
        warnings.warn("torchcrepe not available, using basic note detection")
        return _note_timeline_fallback(audio_path)

    # --- load + onset detection ------------------------------------------
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)

    # --- constants --------------------------------------------------------
    slice_ms = 40  # analysis window
    slice_samps = int(sr * slice_ms / 1000.0)
    hop_len = int(round(16000 * hop_s))

    # --- prepare harmonic layer & CQT for gate ---------------------------
    y_harm, _ = librosa.effects.hpss(y)  # helps both gates
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
            # Polyphony gate (CQT frame centred on onset)
            off = fr + 3
            cqt_frame = C_db[:, off] if off < C_db.shape[1] else C_db[:, -1]
            if _is_polyphonic(cqt_frame, peak_db, min_peaks):
                continue  # reject → chord

            # Periodicity gate (torchcrepe, harmonic layer only)
            start16 = max(0, int(t * 16000) - hop_len//2)
            end16 = start16 + hop_len
            frame = y16[..., start16:end16]  # shape (1, N)
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
            hz = float(f0.squeeze())
            pval = float(pdist.squeeze())
            if pval < conf_thresh or not (80 < hz < 1200):
                continue  # weak/confused

            rows.append((t, librosa.hz_to_note(hz, octave=False)))

    except Exception as e:
        warnings.warn(f"torchcrepe failed ({e}); falling back to pyin")
        for t, fr in zip(onset_times, onset_frames):
            if _is_polyphonic(C_db[:, fr], peak_db, min_peaks):
                continue
            start = max(0, fr*512)
            end = start + slice_samps
            f0, _, _ = librosa.pyin(y[start:end], fmin=80, fmax=1200, sr=sr)
            if f0 is not None and not np.isnan(f0).all():
                hz = float(np.nanmedian(f0))
                rows.append((t, librosa.hz_to_note(hz, octave=False)))

    return pd.DataFrame(rows, columns=["time", "note"])


def _note_timeline_fallback(audio_path: str) -> pd.DataFrame:
    """Basic note detection fallback when torchcrepe is not available."""
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    
    rows = []
    for t in onset_times[:10]:  # Limit to first 10 onsets for performance
        try:
            # Use librosa's pyin for basic pitch detection
            f0, _, _ = librosa.pyin(y, fmin=80, fmax=1200, sr=sr)
            if f0 is not None and not np.isnan(f0).all():
                hz = float(np.nanmedian(f0))
                if 80 < hz < 1200:
                    rows.append((t, librosa.hz_to_note(hz, octave=False)))
        except:
            continue
    
    return pd.DataFrame(rows, columns=["time", "note"])


# --------------------------------------------------------------------------
# Helper Functions
# --------------------------------------------------------------------------
def chord_sequence_by_beat(chords_df: pd.DataFrame, beat_times: np.ndarray):
    """Map beat times to chord labels."""
    idx, seg_i = {}, 0
    for b, bt in enumerate(beat_times):
        while seg_i+1 < len(chords_df) and chords_df.iloc[seg_i]['end'] <= bt:
            seg_i += 1
        idx[b] = chords_df.iloc[seg_i]['label']
    return idx


def convert_to_serializable(obj):
    """Convert pandas DataFrames and NumPy arrays to JSON-serializable types."""
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')
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


# --------------------------------------------------------------------------
# Main Analysis Function
# --------------------------------------------------------------------------
def analyze_guitar_audio(audio_path: str) -> Dict[str, Any]:
    """
    Complete guitar analysis with chords, notes, tempo, and structure.
    This is the modularized version of the original analyse_audio function.
    """
    try:
        y, sr = librosa.load(audio_path, sr=None)
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units='frames', tightness=400)
        tempo = float(np.atleast_1d(tempo)[0])
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        
        # Chord & note timelines (independent)
        chords_df = chord_timeline(audio_path)
        notes_df = note_timeline(audio_path)
        beat_chords = chord_sequence_by_beat(chords_df, beat_times)

        # Onsets / strums (keep original behaviour)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)

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

        # Summarise bars/sections (same as before)
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

        # Convert to serializable format
        result = {
            'tempo_bpm': tempo,
            'strums': [asdict(s) for s in strums],
            'bars': [asdict(b) for b in ordered],
            'chords_timeline': chords_df.to_dict('records'),
            'notes_timeline': notes_df.to_dict('records'),
        }

        return convert_to_serializable(result)

    except Exception as e:
        warnings.warn(f"Analysis failed: {str(e)}")
        # Return minimal fallback structure
        return {
            'tempo_bpm': 120.0,
            'strums': [],
            'bars': [],
            'chords_timeline': [],
            'notes_timeline': [],
            'analysis_metadata': {
                'version': '1.0',
                'method': 'error_fallback',
                'error': str(e)
            }
        }


# --------------------------------------------------------------------------
# Legacy Functions for Compatibility
# --------------------------------------------------------------------------
def detect_time_signature(data: dict) -> dict:
    """Analyze data to determine a likely time signature. Default to 4/4."""
    if 'bars' in data and len(data['bars']) > 0:
        first_bar = data['bars'][0] if isinstance(data['bars'], list) else data['bars'].iloc[0]
        first_pattern = first_bar.get('bit_pattern') if isinstance(first_bar, dict) else None
        if first_pattern:
            length = len(first_pattern)
            if length == 16:
                return {'numerator': 4, 'denominator': 4, 'description': '4/4 time (common time)'}
            if length == 12:
                return {'numerator': 3, 'denominator': 4, 'description': '3/4 time (waltz time)'}
            if length == 20:
                return {'numerator': 5, 'denominator': 4, 'description': '5/4 time'}
            return {'numerator': 4,
                    'denominator': 4,
                    'description': f'Unusual pattern length ({length}), defaulting to 4/4'}
    return {'numerator': 4, 'denominator': 4, 'description': '4/4 time (default)'}


def prepare_data(data_path: str, audio_path: str) -> dict:
    """
    Load JSON analysis results and audio waveform to build visualization payload.
    """
    with open(data_path, 'r') as f:
        data = json.load(f)
    df_strums = pd.DataFrame(data.get('strums', []))
    tempo = float(np.atleast_1d(data.get('tempo_bpm', 0))[0])

    y, sr = librosa.load(audio_path, sr=None)
    duration = len(y) / sr

    hop = 1024
    frame_times = np.arange(0, len(y), hop) / sr
    wave = librosa.util.utils.frame(y, frame_length=hop, hop_length=hop).mean(axis=0)

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)

    strum_intensities = []
    for frame in onset_frames:
        strum_intensities.append(onset_env[frame] if frame < len(onset_env) else 0)
    if strum_intensities:
        max_int = max(strum_intensities)
        if max_int > 0:
            strum_intensities = [i / max_int for i in strum_intensities]

    detected_strums = [{'time': float(t), 'intensity': float(strum_intensities[i] if i < len(strum_intensities) else 0.5)}
                       for i, t in enumerate(onset_times)]

    chord_blocks = []
    if 'chords_timeline' in data:
        for item in data['chords_timeline']:
            chord_blocks.append({'label': item['label'],
                                 'kind': 'CHORD',
                                 'start': float(item['start']),
                                 'end': float(item['end'])})
    else:
        current, start = None, 0
        for i, row in df_strums.iterrows():
            if current is None or row['label'] != current or (i > 0 and row['time'] - df_strums.iloc[i-1]['time'] > 2.0):
                if current is not None:
                    chord_blocks.append({'label': current,
                                          'kind': df_strums.iloc[i-1]['kind'],
                                          'start': start,
                                          'end': row['time']})
                current, start = row['label'], row['time']
        if current is not None:
            chord_blocks.append({'label': current,
                                 'kind': df_strums.iloc[-1]['kind'],
                                 'start': start,
                                 'end': duration})

    note_events = []
    if 'notes_timeline' in data:
        df_notes = pd.DataFrame(data['notes_timeline'])
        current, start = None, 0
        for i, row in df_notes.iterrows():
            t = float(row['time'])
            if current is None or row['note'] != current or (i > 0 and t - float(df_notes.iloc[i-1]['time']) > 0.1):
                if current is not None:
                    note_events.append({'label': current,
                                        'kind': 'NOTE',
                                        'start': start,
                                        'end': t})
                current, start = row['note'], t
        if current is not None:
            note_events.append({'label': current,
                                'kind': 'NOTE',
                                'start': start,
                                'end': min(start + 0.2, duration)})

    sections = []
    if 'bars' in data:
        df_bars = pd.DataFrame(data['bars']) if isinstance(data['bars'], list) else data['bars']
        sec_per_beat = 60 / tempo
        sec_per_bar = sec_per_beat * 4
        start_bar, current_chords = 1, None
        for i, row in df_bars.iterrows():
            bar, chords = row['bar'], row['chords']
            chords_str = ", ".join(chords) if isinstance(chords, list) else str(chords)
            if current_chords is None:
                current_chords = chords_str
            elif chords_str != current_chords:
                sections.append({'start': (start_bar - 1) * sec_per_bar,
                                 'end': (bar - 1) * sec_per_bar,
                                 'start_bar': start_bar,
                                 'end_bar': bar - 1,
                                 'chords': current_chords})
                start_bar, current_chords = bar, chords_str
        if current_chords is not None:
            sections.append({'start': (start_bar - 1) * sec_per_bar,
                             'end': len(df_bars) * sec_per_bar,
                             'start_bar': start_bar,
                             'end_bar': len(df_bars),
                             'chords': current_chords})

    time_signature = detect_time_signature(data)
    return {
        'chord_blocks': chord_blocks,
        'note_events': note_events,
        'sections': sections,
        'detected_strums': detected_strums,
        'strum_data': df_strums.to_dict('records'),
        'tempo': tempo,
        'time_signature': time_signature,
        'waveform': {
            'times': frame_times.tolist(),
            'amplitudes': wave.tolist()
        }
    }