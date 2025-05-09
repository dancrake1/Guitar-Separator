from flask import Flask, render_template, jsonify, send_file
import numpy as np
import pandas as pd
import librosa
import json
import os

# Flask app for serving the visualization
app = Flask(__name__, static_folder='static')

# Load and process data with strum detection from audio
def prepare_data(data_path, audio_path):
    # Load the data (assuming it's stored in a JSON format)
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    # Convert to pandas DataFrames
    df_strums = pd.DataFrame(data['strums'])
    tempo = float(np.atleast_1d(data['tempo_bpm'])[0])
    
    # Process audio for waveform and strum detection
    y, sr = librosa.load(audio_path, sr=None)
    duration = len(y) / sr
    
    # Create downsampled waveform for visualization
    hop = 1024  # Larger hop for smaller data size
    frame_times = np.arange(0, len(y), hop) / sr
    wave = librosa.util.utils.frame(y, frame_length=hop, hop_length=hop).mean(axis=0)
    
    # Detect strums from audio using onset detection
    # This will find moments when new sounds begin
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    
    # Calculate amplitude at each onset for intensity visualization
    strum_intensities = []
    for frame in onset_frames:
        if frame < len(onset_env):
            strum_intensities.append(onset_env[frame])
        else:
            strum_intensities.append(0)
    
    # Normalize intensities to range 0-1
    if len(strum_intensities) > 0:
        max_intensity = max(strum_intensities)
        if max_intensity > 0:
            strum_intensities = [i/max_intensity for i in strum_intensities]
    
    # Create detected strum events
    detected_strums = []
    for i, time in enumerate(onset_times):
        detected_strums.append({
            'time': float(time),
            'intensity': float(strum_intensities[i]) if i < len(strum_intensities) else 0.5
        })
    
    # Create chord blocks (combining consecutive similar chords)
    chord_blocks = []
    current_chord = None
    start_time = 0
    
    for i, row in df_strums.iterrows():
        if current_chord is None or row['label'] != current_chord or (i > 0 and row['time'] - df_strums.iloc[i-1]['time'] > 2.0):
            if current_chord is not None:
                # Add previous block
                end_time = row['time']
                chord_blocks.append({
                    'label': current_chord,
                    'kind': df_strums.iloc[i-1]['kind'],
                    'start': start_time,
                    'end': end_time
                })
            # Start new block
            current_chord = row['label']
            start_time = row['time']
    
    # Add the final block
    if current_chord is not None and len(df_strums) > 0:
        chord_blocks.append({
            'label': current_chord,
            'kind': df_strums.iloc[-1]['kind'],
            'start': start_time,
            'end': duration
        })
    
    # Process sections data
    sections = []
    if 'sections' in data:
        df_sections = pd.DataFrame(data['sections'])
        # Calculate approximate times from bar numbers
        sec_per_beat = 60 / tempo
        sec_per_bar = sec_per_beat * 4  # Assuming 4/4 time
        
        for _, section in df_sections.iterrows():
            start_time = (section['start_bar'] - 1) * sec_per_bar
            end_time = section['end_bar'] * sec_per_bar
            
            sections.append({
                'start': start_time,
                'end': end_time,
                'start_bar': section['start_bar'],
                'end_bar': section['end_bar'],
                'chords': section['chords']
            })
    
    # Detect time signature
    time_signature = detect_time_signature(data)
    
    return {
        'chord_blocks': chord_blocks,
        'sections': sections,
        'detected_strums': detected_strums,  # Audio-detected strums
        'tempo': tempo,
        'time_signature': time_signature,
        'waveform': {
            'times': frame_times.tolist(),
            'amplitudes': wave.tolist()
        },
        'duration': duration
    }

# Time signature detection
def detect_time_signature(data):
    """Analyze the data to determine the likely time signature"""
    
    # Check if bars data exists and has bit patterns
    if 'bars' in data and len(data['bars']) > 0:
        # Get the first bit pattern
        first_bar = data['bars'][0] if isinstance(data['bars'], list) else data['bars'].iloc[0]
        first_pattern = first_bar['bit_pattern'] if 'bit_pattern' in first_bar else None
        
        if first_pattern:
            pattern_length = len(first_pattern)
            
            # Determine time signature based on pattern length
            if pattern_length == 16:
                return {'numerator': 4, 'denominator': 4, 'description': '4/4 time (common time)'}
            elif pattern_length == 12:
                return {'numerator': 3, 'denominator': 4, 'description': '3/4 time (waltz time)'}
            elif pattern_length == 20:
                return {'numerator': 5, 'denominator': 4, 'description': '5/4 time'}
            else:
                return {'numerator': 4, 'denominator': 4, 'description': f'Unusual pattern length ({pattern_length}), defaulting to 4/4'}
    
    # Default to 4/4 if unable to determine
    return {'numerator': 4, 'denominator': 4, 'description': '4/4 time (default)'}

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data')
def get_data():
    return jsonify(visualization_data)

@app.route('/audio')
def get_audio():
    return send_file(audio_path, mimetype='audio/wav')

if __name__ == '__main__':
    # Configuration - update these paths
    data_path = 'testing/outputs/JeffBuckley-LoverYouShouldveComeOverAudio/guitar_data.json'  
    audio_path = 'testing/outputs/JeffBuckley-LoverYouShouldveComeOverAudio/stage2_guitar_enhanced_cut.wav'
    
    # Prepare the visualization data
    visualization_data = prepare_data(data_path, audio_path)
    
    # Create static folder if it doesn't exist
    os.makedirs('static', exist_ok=True)
    
    # Run the app
    app.run(debug=True) 