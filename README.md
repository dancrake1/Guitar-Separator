# Guitar Separator

A Flask web application that separates guitar tracks from audio files and provides real-time chord and note detection with interactive visualization.

## Features

- **Two-Stage AI Separation**: Advanced guitar isolation using Demucs models
- **Real-Time Processing**: Stream processing updates with live progress feedback
- **Interactive Visualization**: Dynamic chord and note analysis with audio playback
- **Time-Based Trimming**: Process specific sections of audio files
- **Multiple Audio Formats**: Support for MP3, WAV, and other common formats

## Architecture

The application uses a sophisticated two-stage processing pipeline:

1. **Stage 1**: Extract the 'other' stem using `htdemucs_ft` model
2. **Stage 2**: Isolate and enhance guitar from the 'other' stem using `htdemucs_6s` model
3. **Analysis**: Guitar transcription and chord detection using FretNet model

## Installation

### Prerequisites

- Python 3.8+
- FFmpeg
- CUDA-compatible GPU (recommended for faster processing)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd Guitar-Separator
```

2. Create and activate a virtual environment:
```bash
python -m venv separator_env
source separator_env/bin/activate  # On Windows: separator_env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Additional Dependencies

For optimal performance, install PyTorch with CUDA support:
```bash
# Install specific PyTorch version
pip install torch==2.7.0 torchvision

# Install torchaudio with FFmpeg support (macOS)
CMAKE_PREFIX_PATH=$(brew --prefix ffmpeg@6) pip install --no-binary torchaudio torchaudio

# Install guitar transcription models
pip install --no-deps git+https://github.com/cwitkowitz/guitar-transcription-continuous
pip install --no-deps --upgrade git+https://github.com/cwitkowitz/amt-tools
```

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your browser and navigate to `http://localhost:5000`

3. Upload an audio file and select processing options:
   - **Mode**: Choose between full audio or trimmed section
   - **Time Range**: Specify start and end times (optional)

4. Monitor real-time processing progress and view results with interactive visualization

## API Endpoints

- `GET /` - Main application interface
- `POST /process` - Synchronous audio processing
- `POST /process_stream` - Streaming audio processing with live updates
- `GET /data` - Retrieve visualization data
- `GET /audio?track=<guitar|original>` - Download processed audio

## Project Structure

```
Guitar-Separator/
├── app.py                 # Flask application and routes
├── utils/                 # Core processing utilities
│   ├── stage1_utils.py    # First-stage separation (htdemucs_ft)
│   ├── stage2_utils.py    # Second-stage separation (htdemucs_6s)
│   ├── analysis_utils.py  # Guitar analysis and visualization data
│   ├── pipeline_utils.py  # Processing pipeline orchestration
│   ├── file_utils.py      # File and session management
│   └── trim_utils.py      # Audio trimming utilities
├── testing/
│   ├── testing_split.py   # Main audio processing pipeline script
│   └── __init__.py        # Python package marker
├── templates/
│   └── index.html         # Web interface
├── FretNet/
│   └── models/            # Guitar transcription model
└── static/
    └── uploads/           # Session-based file storage (created at runtime)
```

## Model Details

- **Demucs Models**: Meta's state-of-the-art source separation
  - `htdemucs_ft`: Fine-tuned hybrid transformer model
  - `htdemucs_6s`: 6-source separation model
- **FretNet**: Guitar transcription and chord detection model

## Development

### Future Enhancements

- Enhanced chord recognition and visualization
- Guitar chord shape images and fingering diagrams
- Additional audio export formats
- Batch processing capabilities

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Requirements

See `requirements.txt` for complete dependency list. Key libraries include:

- Flask (web framework)
- PyTorch (deep learning models)
- Demucs (source separation)
- Librosa (audio processing)
- NumPy, SciPy (numerical computing)

## License

[Add your license information here]

## Acknowledgments

- [Demucs](https://github.com/facebookresearch/demucs) for source separation models
- [FretNet](https://github.com/cwitkowitz/guitar-transcription-continuous) for guitar transcription
- Meta AI Research for the htdemucs models