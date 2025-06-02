from flask import Flask, render_template, jsonify, send_file, request, redirect, url_for, abort, Response, stream_with_context
import numpy as np
import pandas as pd
import librosa
import subprocess
import sys
import uuid
import torchaudio
from werkzeug.utils import secure_filename
import json
import os

# Flask app for serving the visualization
app = Flask(__name__, static_folder='static')

# Globals to hold current processing state
visualization_data = None
guitar_audio_path = None
original_audio_path = None

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
    
    # Process chord timeline data (use direct chords_timeline if available)
    chord_blocks = []
    if 'chords_timeline' in data:
        for item in data['chords_timeline']:
            chord_blocks.append({
                'label': item['label'],
                'kind': 'CHORD',  # All items in chords_timeline are chords
                'start': float(item['start']),
                'end': float(item['end'])
            })
    else:
        # Fallback to original method if chords_timeline not available
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
    
    # Process individual notes timeline if available
    note_events = []
    if 'notes_timeline' in data:
        df_notes = pd.DataFrame(data['notes_timeline'])
        
        # Group consecutive identical notes
        current_note = None
        start_time = 0
        
        for i, row in df_notes.iterrows():
            if current_note is None or row['note'] != current_note or (i > 0 and float(row['time']) - float(df_notes.iloc[i-1]['time']) > 0.1):
                if current_note is not None:
                    # Add previous note block
                    end_time = float(row['time'])
                    note_events.append({
                        'label': current_note,
                        'kind': 'NOTE',
                        'start': start_time,
                        'end': end_time
                    })
                # Start new note
                current_note = row['note']
                start_time = float(row['time'])
                
        # Add the final note
        if current_note is not None and len(df_notes) > 0:
            note_events.append({
                'label': current_note,
                'kind': 'NOTE',
                'start': start_time,
                'end': min(start_time + 0.2, duration)  # Assume note lasts at most 0.2s if it's the last one
            })
    
    # Process sections data
    sections = []
    if 'bars' in data:
        df_bars = pd.DataFrame(data['bars']) if isinstance(data['bars'], list) else data['bars']
        # Calculate approximate times from bar numbers
        sec_per_beat = 60 / tempo
        sec_per_bar = sec_per_beat * 4  # Assuming 4/4 time
        
        start_bar = 1
        current_chords = None
        
        for i, row in df_bars.iterrows():
            bar_num = row['bar']
            chords = row['chords']
            
            # Convert chords to string for comparison if it's a list
            chords_str = ", ".join(chords) if isinstance(chords, list) else str(chords)
            
            if current_chords is None:
                current_chords = chords_str
            elif chords_str != current_chords:
                # End previous section
                end_bar = bar_num - 1
                
                sections.append({
                    'start': (start_bar - 1) * sec_per_bar,
                    'end': end_bar * sec_per_bar,
                    'start_bar': start_bar,
                    'end_bar': end_bar,
                    'chords': current_chords
                })
                
                # Start new section
                start_bar = bar_num
                current_chords = chords_str
        
        # Add the final section
        if current_chords is not None:
            sections.append({
                'start': (start_bar - 1) * sec_per_bar,
                'end': len(df_bars) * sec_per_bar,
                'start_bar': start_bar,
                'end_bar': len(df_bars),
                'chords': current_chords
            })
    
    # Detect time signature
    time_signature = detect_time_signature(data)
    
    return {
        'chord_blocks': chord_blocks,
        'note_events': note_events,
        'sections': sections,
        'detected_strums': detected_strums,  # Audio-detected strums
        'strum_data': df_strums.to_dict('records'),  # Include original strum data
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


@app.route('/process', methods=['POST'])
def process():
    """Handle uploaded audio, run separation/analysis pipeline, and prepare data for visualization."""
    global visualization_data, guitar_audio_path, original_audio_path
    # Validate upload
    uploaded = request.files.get('audio_file')
    if not uploaded:
        abort(400, 'No audio file uploaded')
    # Parse form fields
    mode = request.form.get('mode', 'full')
    try:
        start_sec = float(request.form.get('start_sec', 0) or 0)
    except ValueError:
        start_sec = 0.0
    end_val = request.form.get('end_sec')
    try:
        end_sec = float(end_val) if end_val else None
    except ValueError:
        end_sec = None

    # Create unique session folder under static/uploads
    session_id = uuid.uuid4().hex
    base_dir = os.path.join(app.static_folder, 'uploads', session_id)
    os.makedirs(base_dir, exist_ok=True)

    # Save uploaded file
    filename = secure_filename(uploaded.filename)
    input_path = os.path.join(base_dir, filename)
    uploaded.save(input_path)

    # Run separation+analysis script
    script_path = os.path.join(os.path.dirname(__file__), 'testing', 'testing_split.py')
    # Build command for external processing script
    cmd = [sys.executable, script_path, input_path,
           '--start-sec', str(start_sec),
           '--output-dir', base_dir]
    if end_sec is not None:
        cmd.extend(['--end-sec', str(end_sec)])
    # Execute external processing script
    subprocess.run(cmd, check=True)

    # Determine song_name directory
    song_name = os.path.splitext(os.path.basename(input_path))[0].replace('_', '')
    subdir = os.path.join(base_dir, song_name)

    # Paths to generated files
    data_json = os.path.join(subdir, 'guitar_data.json')
    guitar_file = os.path.join(subdir, 'stage2_guitar_enhanced_cut.wav')

    # Trim original if requested
    if mode == 'trimmed':
        # Trim the uploaded original audio
        waveform, sr = torchaudio.load(input_path)
        start_sample = int(start_sec * sr)
        end_sample = int(end_sec * sr) if end_sec is not None else waveform.shape[1]
        trimmed_waveform = waveform[:, start_sample:end_sample]
        orig_out = os.path.join(subdir, 'original_trimmed.wav')
        torchaudio.save(orig_out, trimmed_waveform, sr)
        original_file = orig_out
    else:
        original_file = input_path

    # Prepare visualization data
    visualization_data = prepare_data(data_json, guitar_file)
    guitar_audio_path = guitar_file
    original_audio_path = original_file

    return redirect(url_for('index'))

@app.route('/process_stream', methods=['POST'])
def process_stream():
    """Stream processing steps for uploaded audio via server-sent progress."""
    global visualization_data, guitar_audio_path, original_audio_path

    def generate():
        uploaded = request.files.get('audio_file')
        if not uploaded:
            yield "Error: no audio file uploaded\n"
            return
        mode = request.form.get('mode', "full")
        try:
            start_sec = float(request.form.get("start_sec", 0) or 0)
        except ValueError:
            start_sec = 0.0
        end_val = request.form.get("end_sec")
        try:
            end_sec = float(end_val) if end_val else None
        except ValueError:
            end_sec = None

        yield f"Saving upload ({uploaded.filename})...\n"
        session_id = uuid.uuid4().hex
        base_dir = os.path.join(app.static_folder, "uploads", session_id)
        os.makedirs(base_dir, exist_ok=True)
        filename = secure_filename(uploaded.filename)
        input_path = os.path.join(base_dir, filename)
        uploaded.save(input_path)
        yield f"Uploaded to {input_path}\n"

        yield "Running separation and analysis...\n"
        script_path = os.path.join(os.path.dirname(__file__), "testing", "testing_split.py")
        cmd = [sys.executable, script_path, input_path, "--start-sec", str(start_sec), "--output-dir", base_dir]
        if end_sec is not None:
            cmd.extend(["--end-sec", str(end_sec)])
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout or []:
            yield line
        proc.wait()
        if proc.returncode != 0:
            yield f"Error: processing script failed with code {proc.returncode}\n"
            return
        yield "Separation and analysis complete.\n"

        song_name = os.path.splitext(os.path.basename(input_path))[0].replace("_", "")
        subdir = os.path.join(base_dir, song_name)
        data_json = os.path.join(subdir, "guitar_data.json")
        guitar_file = os.path.join(subdir, "stage2_guitar_enhanced_cut.wav")

        if mode == "trimmed":
            yield "Trimming original audio...\n"
            waveform, sr = torchaudio.load(input_path)
            start_sample = int(start_sec * sr)
            end_sample = int(end_sec * sr) if end_sec is not None else waveform.shape[1]
            trimmed_waveform = waveform[:, start_sample:end_sample]
            orig_out = os.path.join(subdir, "original_trimmed.wav")
            torchaudio.save(orig_out, trimmed_waveform, sr)
            original_file = orig_out
            yield "Original audio trimmed.\n"
        else:
            yield "Keeping full original audio.\n"
            original_file = input_path

        yield "Preparing visualization data...\n"
        visualization_data = prepare_data(data_json, guitar_file)
        guitar_audio_path = guitar_file
        original_audio_path = original_file
        yield "Done processing.\n"

    return Response(stream_with_context(generate()), mimetype="text/plain")

@app.route('/data')
def get_data():
    if visualization_data is None:
        return jsonify({})
    return jsonify(visualization_data)

@app.route('/audio')
def get_audio():
    if not guitar_audio_path or not original_audio_path:
        abort(404)
    track = request.args.get('track', 'guitar')
    if track == 'original':
        path = original_audio_path
    else:
        path = guitar_audio_path
    return send_file(path, mimetype='audio/wav')

if __name__ == '__main__':
    # Ensure uploads directory exists
    os.makedirs(os.path.join(app.static_folder, 'uploads'), exist_ok=True)
    app.run(debug=True)