from flask import Flask, render_template, jsonify, send_file, request, redirect, url_for, abort, Response, stream_with_context
import os
import logging


from utils.file_utils import make_session_folder, save_uploaded_file, make_song_subfolder
from utils.audio_processor import process_audio_file
from utils.trim_utils import trim_audio

# Flask app for serving the visualization
app = Flask(__name__, static_folder='static')

# Globals to hold current processing state
visualization_data = None
guitar_audio_path = None
original_audio_path = None

# Routes
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process():
    """Handle uploaded audio synchronously (fallback) and redirect back to index."""
    global visualization_data, guitar_audio_path, original_audio_path
    from utils.analysis_utils import prepare_data
    uploaded = request.files.get('audio_file')
    if not uploaded:
        abort(400, 'No audio file uploaded')
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

    session_id, base_dir = make_session_folder(app.static_folder)
    filename, input_path = save_uploaded_file(uploaded, base_dir)
    song_name = os.path.splitext(filename)[0].replace('_', '')
    subdir = make_song_subfolder(base_dir, song_name)

    # Run the separation & analysis pipeline
    for _ in process_audio_file(input_path, base_dir, start_sec, end_sec):
        pass

    # Paths to results
    data_json = os.path.join(subdir, 'guitar_data.json')
    guitar_file = os.path.join(subdir, 'stage2_guitar_enhanced_cut.wav')

    # Check if trimmed original was created by the processor
    original_trimmed_file = os.path.join(subdir, 'original_trimmed.wav')
    if os.path.exists(original_trimmed_file):
        original_file = original_trimmed_file
    else:
        original_file = input_path

    visualization_data = prepare_data(data_json, guitar_file)
    guitar_audio_path = guitar_file
    original_audio_path = original_file
    return redirect(url_for('index'))

@app.route('/process_stream', methods=['POST'])
def process_stream():
    """Stream processing steps for uploaded audio via server-sent progress."""
    global visualization_data, guitar_audio_path, original_audio_path

    def generate():
        global visualization_data, guitar_audio_path, original_audio_path
        from utils.analysis_utils import prepare_data
        uploaded = request.files.get('audio_file')
        if not uploaded:
            yield "Error: no audio file uploaded\n"
            return
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

        # Prepare session folder and per-session logger
        session_id, base_dir = make_session_folder(app.static_folder)
        log_dir = os.path.abspath('logs')
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"{session_id}.log")
        logger = logging.getLogger(session_id)
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(log_path)
        fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        logger.addHandler(fh)

        yield f"Saving upload ({uploaded.filename})...\n"
        logger.info(f"Saving upload ({uploaded.filename})")
        filename, input_path = save_uploaded_file(uploaded, base_dir)
        yield f"Uploaded to {input_path}\n"
        logger.info(f"Uploaded to {input_path}")

        yield "Running separation and analysis...\n"
        logger.info("Running separation and analysis")
        for line in process_audio_file(input_path, base_dir, start_sec, end_sec):
            yield f"{line}\n"
            logger.info(line)
        yield "Separation and analysis complete.\n"
        logger.info("Separation and analysis complete.")

        song_name = os.path.splitext(filename)[0].replace('_', '')
        subdir = make_song_subfolder(base_dir, song_name)
        data_json = os.path.join(subdir, 'guitar_data.json')
        guitar_file = os.path.join(subdir, 'stage2_guitar_enhanced_cut.wav')

        # Check if trimmed original was created by the processor
        original_trimmed_file = os.path.join(subdir, 'original_trimmed.wav')
        if os.path.exists(original_trimmed_file):
            original_file = original_trimmed_file
            yield "Using processor-trimmed original audio.\n"
            logger.info("Using processor-trimmed original audio")
        else:
            original_file = input_path
            yield "Using full original audio.\n"
            logger.info("Using full original audio")

        yield "Preparing visualization data...\n"
        logger.info("Preparing visualization data")
        visualization_data = prepare_data(data_json, guitar_file)
        guitar_audio_path = guitar_file
        original_audio_path = original_file
        yield "Done processing.\n"
        logger.info("Done processing.")

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