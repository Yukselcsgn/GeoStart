import os
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import logging


@dataclass
class Config:
    # API Configuration
    API_KEY: str = os.getenv("GMAPS_API_KEY", "Your Api Key")
    STREETVIEW_URL: str = "https://maps.googleapis.com/maps/api/streetview"
    PLACES_URL: str = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    PLACE_PHOTOS_URL: str = "https://maps.googleapis.com/maps/api/place/photo"
    PLACE_DETAILS_URL: str = "https://maps.googleapis.com/maps/api/place/details/json"
    MAX_PHOTOS_PER_PLACE: int = 5

    # Directory Configuration
    BASE_DIR: Path = Path(__file__).parent.parent
    IMAGE_DIR: Path = BASE_DIR / "images"
    METADATA_FILE: Path = BASE_DIR / "metadata.csv"
    LOG_FILE: Path = BASE_DIR / "download_log.log"
    ROAD_NETWORK_FILE: Path = BASE_DIR / "road_network.geojson"

    # Image Parameters
    IMG_WIDTH: int = 1280  # Higher resolution for better ML results
    IMG_HEIGHT: int = 1280
    FOV: int = 90  # Field of view
    PITCH_RANGE: tuple = (-10, 30)  # Min, max pitch for varied angles
    HEADING_STEP: int = 30  # Degrees between each 360° photo

    # Download Parameters
    MAX_RETRIES: int = 5
    RETRY_DELAY: float = 2.5  # seconds
    REQUEST_TIMEOUT: int = 15  # seconds
    RATE_LIMIT_DELAY: float = 0.1  # seconds between API calls

    # Area Coverage
    MIN_DISTANCE_BETWEEN_POINTS: float = 20.0  # meters
    ROAD_SAMPLING_INTERVAL: float = 50.0  # meters

    def validate(self):
        """Validate configuration parameters"""
        if not self.API_KEY or self.API_KEY.startswith("YOUR_API_KEY"):
            raise ValueError("Invalid Google Maps API Key")

        if not self.BASE_DIR.exists():
            raise FileNotFoundError(f"Base directory does not exist: {self.BASE_DIR}")

        if self.IMG_WIDTH < 640 or self.IMG_HEIGHT < 640:
            raise ValueError("Image dimensions too small (minimum 640x640)")

        if self.FOV < 10 or self.FOV > 120:
            raise ValueError("FOV must be between 10 and 120 degrees")


config = Config()

try:
    config.validate()
except Exception as e:
    logging.error(f"Configuration error: {str(e)}")
    raise