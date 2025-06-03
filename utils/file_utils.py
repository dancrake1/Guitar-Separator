"""
File- and session-related utilities for uploads.
"""
import os
import uuid
from werkzeug.utils import secure_filename


def make_session_folder(static_folder: str) -> tuple[str, str]:
    """
    Create a unique session upload folder under the given static_folder.

    Returns:
        session_id: The UUID for this session.
        base_dir: Full path to the session folder.
    """
    session_id = uuid.uuid4().hex
    base_dir = os.path.join(static_folder, 'uploads', session_id)
    os.makedirs(base_dir, exist_ok=True)
    return session_id, base_dir


def save_uploaded_file(uploaded_file, base_dir: str) -> tuple[str, str]:
    """
    Securely save the uploaded file into base_dir.

    Returns:
        filename: The secure filename used.
        path: Full path to the saved file.
    """
    filename = secure_filename(uploaded_file.filename)
    path = os.path.join(base_dir, filename)
    uploaded_file.save(path)
    return filename, path


def make_song_subfolder(base_dir: str, song_name: str) -> str:
    """
    Create and return a subdirectory for a given song under base_dir.
    """
    subdir = os.path.join(base_dir, song_name)
    os.makedirs(subdir, exist_ok=True)
    return subdir