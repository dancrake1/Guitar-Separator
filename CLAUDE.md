# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Guitar Separator is a Flask web application that separates guitar tracks from audio files and provides chord/note detection and visualization. It uses a two-stage AI pipeline: first using Demucs for source separation, then FretNet for guitar analysis.

## File Handling Guidelines

**NEVER read or process binary audio files (.mp3, .wav) - they are large assets that will consume context unnecessarily.**
- Skip all files in `testing/outputs/`, `static/uploads/` directories  
- Focus only on source code files (.py, .html, .js, .json, .md)
- When analyzing project structure, exclude media files from consideration

## Development Setup

Install dependencies using the commands in `info.txt`:
```bash
pip install -U pip wheel setuptools
pip install torch==2.7.0 torchvision
pip install "soundfile>=0.12,<1.0"
CMAKE_PREFIX_PATH=$(brew --prefix ffmpeg@6) pip install --no-binary torchaudio torchaudio
pip install librosa numpy scipy tqdm matplotlib jupyterlab ffmpeg-python demucs
pip install madmom torchcrepe pandas
pip install --no-deps git+https://github.com/cwitkowitz/guitar-transcription-continuous
pip install --no-deps --upgrade git+https://github.com/cwitkowitz/amt-tools
```

Run the Flask application:
```bash
python app.py
```

## Architecture

The application follows a two-stage processing pipeline:

### Core Processing Pipeline
- **Stage 1** (`utils/stage1_utils.py`): Uses htdemucs_ft model to extract the 'other' stem from mixed audio
- **Stage 2** (`utils/stage2_utils.py`): Uses htdemucs_6s model to extract and enhance guitar from the 'other' stem
- **Analysis** (`utils/analysis_utils.py`): Processes separated guitar audio for chord/note detection and visualization data

### Web Interface
- **Flask App** (`app.py`): Serves the web interface with synchronous (`/process`) and streaming (`/process_stream`) endpoints
- **Template** (`templates/index.html`): Single-page interface for audio upload and visualization
- **Static Files** (`static/uploads/`): Stores uploaded files and processing results organized by session ID

### Utility Modules
- `utils/file_utils.py`: Handles session folders, file uploads, and directory management
- `utils/pipeline_utils.py`: Runs the external processing script (`testing/testing_split.py`) as subprocess
- `utils/trim_utils.py`: Audio trimming functionality for time-based selections

### Processing Script
- `testing/testing_split.py`: Main processing script that orchestrates the two-stage separation pipeline

## Key Dependencies

- **Demucs**: Source separation models (htdemucs_ft, htdemucs_6s)
- **FretNet**: Guitar transcription model (`FretNet/models/fold-0/model-2000.pt`)
- **Librosa/Torchaudio**: Audio processing and analysis
- **Flask**: Web server and API endpoints

## File Organization

Processed files are organized by session ID under `static/uploads/{session_id}/`:
- Original uploaded files
- `{song_name}/stage1_other.wav`: Output from stage 1 separation
- `{song_name}/stage2_guitar_enhanced.wav`: Enhanced guitar track
- `{song_name}/guitar_data.json`: Analysis results for visualization

Logs are stored in `logs/{session_id}.log` for debugging processing steps.

## Important Notes

- The pipeline requires GPU support for optimal performance with PyTorch models
- Session management uses unique IDs to isolate concurrent processing requests
- Virtual environment is in `seperator_env/` directory