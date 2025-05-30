import os
import time
import logging
import math
import json
import pandas as pd
from typing import List, Tuple, Dict, Optional, Union
from pathlib import Path
from geopy.distance import geodesic
from PIL import Image, ImageOps
from io import BytesIO
import requests
from requests.exceptions import RequestException
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from data.config import config  # Config sınıfının örneğini almış oluyorsun

logger = logging.getLogger(__name__)


def setup_logging(log_file: Path) -> logging.Logger:
    """Configure comprehensive logging system"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='a', encoding='utf-8'),
            logging.StreamHandler(),
        ]
    )
    return logging.getLogger(__name__)


def ensure_directory_exists(directory: Path) -> None:
    """Ensure directory exists with proper permissions"""
    try:
        directory.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Directory ensured: {directory}")
    except (PermissionError, OSError) as e:
        logger.error(f"Failed to create directory {directory}: {str(e)}")
        raise


def save_image(image_data: bytes, file_path: Path, quality: int = 90) -> bool:
    """Save image with validation and error handling"""
    try:
        with Image.open(BytesIO(image_data)) as img:
            # Auto-rotate based on EXIF data
            img = ImageOps.exif_transpose(img)
            img.save(file_path, quality=quality, optimize=True)
        logger.debug(f"Image saved successfully: {file_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save image {file_path}: {str(e)}")
        return False


def save_metadata(metadata: List[dict], file_path: Path) -> bool:
    """Save metadata with atomic write operation"""
    try:
        temp_file = file_path.with_suffix('.tmp')

        if file_path.suffix == '.csv':
            df = pd.DataFrame(metadata)
            df.to_csv(temp_file, index=False)
        elif file_path.suffix == '.json':
            with open(temp_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        else:
            raise ValueError("Unsupported file format")

        # Atomic rename
        temp_file.replace(file_path)
        logger.info(f"Metadata saved successfully: {file_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save metadata: {str(e)}")
        if temp_file.exists():
            temp_file.unlink()
        return False


def calculate_distance(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
    """Calculate geodesic distance in meters with high precision"""
    try:
        return geodesic(coord1, coord2).meters
    except ValueError as e:
        logger.error(f"Invalid coordinates: {coord1} or {coord2} - {str(e)}")
        raise


def make_api_request(url: str, params: dict, session: Optional[requests.Session] = None) -> Optional[requests.Response]:
    """Robust API request with retry logic and rate limiting"""
    for attempt in range(config.MAX_RETRIES):
        try:
            req_fn = session.get if session else requests.get
            response = req_fn(
                url,
                params=params,
                timeout=config.REQUEST_TIMEOUT,
                headers={'User-Agent': 'StreetViewDownloader/1.0'}
            )

            if response.status_code == 200:
                return response
            elif response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 10))
                logger.warning(f"Rate limited. Waiting {retry_after} seconds...")
                time.sleep(retry_after)
                continue

            logger.warning(f"Attempt {attempt + 1}: Status {response.status_code}")
        except RequestException as e:
            logger.warning(f"Attempt {attempt + 1}: Request failed - {str(e)}")

        if attempt < config.MAX_RETRIES - 1:
            time.sleep(config.RETRY_DELAY * (attempt + 1))

    logger.error(f"Max retries exceeded for URL: {url}")
    return None


def generate_grid_points(center: Tuple[float, float], radius_meters: float) -> List[Tuple[float, float]]:
    """Generate systematic grid points around center with proper geodetic calculations"""
    earth_radius = 6378137  # meters
    lat, lon = center

    # Convert radius from meters to degrees (approximate)
    lat_delta = (radius_meters / earth_radius) * (180 / math.pi)
    lon_delta = (radius_meters / (earth_radius * math.cos(math.pi * lat / 180))) * (180 / math.pi)

    steps = int(radius_meters / config.MIN_DISTANCE_BETWEEN_POINTS)
    points = []

    for i in np.linspace(-1, 1, steps):
        for j in np.linspace(-1, 1, steps):
            # Only include points within circular radius
            if i ** 2 + j ** 2 <= 1:
                new_lat = lat + i * lat_delta
                new_lon = lon + j * lon_delta
                points.append((new_lat, new_lon))

    return points


def parallel_execute(func, items: list, max_workers: int = 4) -> list:
    """Execute function in parallel with thread pooling"""
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(func, item): item for item in items}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                logger.error(f"Task failed: {str(e)}")
    return results