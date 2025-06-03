"""
Analysis utilities: prepare data for visualization and detect time signature.
"""
import json
import numpy as np
import pandas as pd
import librosa


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