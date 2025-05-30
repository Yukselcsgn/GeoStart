"""
StreetView Image Downloader Package

A comprehensive tool for downloading street-level imagery and place photos from Google Maps API,
with systematic coverage of road networks and points of interest.
"""

from .downloader import StreetViewDownloader
from .places import PlacePhotoDownloader
from .roads import RoadNetworkDownloader
from .config import config
from .utils import setup_logging

# Initialize logging
setup_logging(config.LOG_FILE)

__version__ = "1.0.0"
__all__ = [
    'StreetViewDownloader',
    'PlacePhotoDownloader',
    'RoadNetworkDownloader',
    'config'
]

# Package-level documentation
__doc__ = """
StreetView Image Downloader

Features:
- Systematic StreetView image collection
- 360° panorama capture
- Place photo collection (businesses, landmarks)
- Road network analysis
- Metadata management
- Parallel downloading
"""