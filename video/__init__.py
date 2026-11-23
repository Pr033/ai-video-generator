"""Video generation backend providers and helpers."""
from .base import VideoProvider, VideoSubmissionError, api_call, download_file
from .frames import extract_first_frame, extract_last_frame, image_to_base64, check_ffmpeg
from .seedance import SeedanceProvider, MiniMaxProvider

__all__ = [
    "VideoProvider", "VideoSubmissionError", "api_call", "download_file",
    "extract_first_frame", "extract_last_frame", "image_to_base64", "check_ffmpeg",
    "SeedanceProvider", "MiniMaxProvider",
]
