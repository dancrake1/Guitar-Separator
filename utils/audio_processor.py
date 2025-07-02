"""
Main audio processing orchestrator - consolidates all separation and analysis logic.
"""
import os
import json
import torch
import torchaudio
import numpy as np
import librosa
from typing import Optional, Iterator, Tuple, Dict, Any

from .stage1_utils import run_stage1
from .stage2_utils import run_stage2
from .analysis_utils import analyze_guitar_audio
from .file_utils import make_song_subfolder


class AudioProcessor:
    """Main audio processor that orchestrates the two-stage separation pipeline."""
    
    def __init__(self):
        self.device = self._get_device()
        
    def _get_device(self) -> torch.device:
        """Determine the best available device for processing."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    def process_audio(self,
                     input_path: str,
                     output_dir: str,
                     start_sec: float = 0.0,
                     end_sec: Optional[float] = None,
                     progress_callback: Optional[callable] = None) -> Iterator[str]:
        """
        Process audio file through the complete pipeline.
        
        Args:
            input_path: Path to input audio file
            output_dir: Base output directory
            start_sec: Start time for trimming (seconds)
            end_sec: End time for trimming (seconds, None for end of file)
            progress_callback: Optional callback for progress updates
            
        Yields:
            Progress messages
        """
        try:
            # Setup output directory
            song_name = os.path.splitext(os.path.basename(input_path))[0].replace('_', '')
            song_dir = make_song_subfolder(output_dir, song_name)
            
            yield f"Processing {song_name}..."
            yield f"Using device: {self.device}"
            
            # Load and trim audio
            yield "Loading audio file..."
            waveform, sample_rate = torchaudio.load(input_path)
            yield f"Loaded audio with shape {waveform.shape} and sample rate {sample_rate}"
            
            # Store reference to potentially trimmed input for original audio alignment
            trimmed_input_path = input_path
            if start_sec > 0 or end_sec is not None:
                yield f"Trimming audio: {start_sec}s to {end_sec or 'end'}s"
                start_sample = int(start_sec * sample_rate)
                end_sample = int(end_sec * sample_rate) if end_sec is not None else waveform.shape[1]
                waveform = waveform[:, start_sample:end_sample]
                yield f"Trimmed audio shape: {waveform.shape}"
                
                # Save trimmed input for perfect original audio alignment
                trimmed_input_path = os.path.join(song_dir, "input_trimmed.wav")
                torchaudio.save(trimmed_input_path, waveform, sample_rate)
                yield f"Saved trimmed input audio to {trimmed_input_path}"
            
            # Stage 1: Extract 'other' stem
            yield "STAGE 1: Separating with htdemucs_ft..."
            other_waveform, stage1_sr = run_stage1(waveform, sample_rate, self.device)
            
            # Save stage1 output
            other_file = os.path.join(song_dir, "stage1_other.wav")
            torchaudio.save(other_file, other_waveform, stage1_sr)
            yield f"Saved 'other' stem to {other_file}"
            
            # Stage 2: Extract guitar from 'other' stem
            yield "STAGE 2: Extracting guitar from 'other' using htdemucs_6s..."
            guitar_waveform, enhanced_guitar, stage2_sr = run_stage2(other_waveform, stage1_sr, self.device)
            
            # Save raw guitar extraction
            guitar_file = os.path.join(song_dir, "stage2_guitar_from_other.wav")
            torchaudio.save(guitar_file, guitar_waveform, stage2_sr)
            yield f"Saved extracted guitar to {guitar_file}"
            
            # Save enhanced guitar (already processed in run_stage2)
            yield "Saving enhanced guitar audio..."
            enhanced_file = os.path.join(song_dir, "stage2_guitar_enhanced.wav")
            torchaudio.save(enhanced_file, enhanced_guitar, stage2_sr)
            yield f"Saved enhanced guitar to {enhanced_file}"
            
            # Set up final output files - no additional trimming needed since we processed trimmed audio
            import shutil
            trimmed_file = os.path.join(song_dir, "stage2_guitar_enhanced_cut.wav")
            shutil.copy2(enhanced_file, trimmed_file)
            yield f"Using enhanced guitar as final output: {trimmed_file}"
            
            # Use the trimmed input (if trimming was requested) for perfect alignment
            if start_sec > 0 or end_sec is not None:
                yield "Using pre-trimmed original audio for perfect alignment..."
                original_trimmed = os.path.join(song_dir, "original_trimmed.wav")
                shutil.copy2(trimmed_input_path, original_trimmed)
                yield f"Aligned original audio saved to {original_trimmed}"
            
            # Guitar analysis
            yield "Analyzing guitar for chords and notes..."
            analysis_data = analyze_guitar_audio(trimmed_file)
            
            # Save analysis data
            data_file = os.path.join(song_dir, "guitar_data.json")
            with open(data_file, 'w') as f:
                json.dump(analysis_data, f, indent=2, default=self._json_serializer)
            yield f"Saved analysis data to {data_file}"
            
            yield "Processing complete!"
            
        except Exception as e:
            yield f"Error during processing: {str(e)}"
            raise
    
    def _json_serializer(self, obj):
        """Custom JSON serializer for numpy types."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def process_audio_file(input_path: str,
                      output_dir: str,
                      start_sec: float = 0.0,
                      end_sec: Optional[float] = None,
                      progress_callback: Optional[callable] = None) -> Iterator[str]:
    """
    Convenience function to process an audio file.
    
    Args:
        input_path: Path to input audio file
        output_dir: Base output directory
        start_sec: Start time for trimming (seconds)
        end_sec: End time for trimming (seconds, None for end of file)
        progress_callback: Optional callback for progress updates
        
    Yields:
        Progress messages
    """
    processor = AudioProcessor()
    yield from processor.process_audio(input_path, output_dir, start_sec, end_sec, progress_callback)