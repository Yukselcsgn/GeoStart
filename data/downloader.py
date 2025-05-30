import os
import time
import logging
import math
from typing import List, Tuple, Dict, Optional
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import random
import requests

from .config import config
from .utils import (
    setup_logging,
    ensure_directory_exists,
    save_image,
    save_metadata,
    make_api_request,
    calculate_distance,
    parallel_execute, generate_grid_points
)

logger = setup_logging(config.LOG_FILE)


class StreetViewDownloader:
    def __init__(self, session: Optional[requests.Session] = None):
        ensure_directory_exists(config.IMAGE_DIR)
        self.metadata = []
        self.session = session or requests.Session()
        self.last_request_time = 0

    def _rate_limit(self):
        """Enforce rate limiting between API calls"""
        elapsed = time.time() - self.last_request_time
        if elapsed < config.RATE_LIMIT_DELAY:
            time.sleep(config.RATE_LIMIT_DELAY - elapsed)
        self.last_request_time = time.time()

    def _generate_pano_params(self, lat: float, lon: float) -> List[Dict]:
        """Generate parameters for 360° panorama capture"""
        params_list = []
        base_filename = f"sv_{lat:.6f}_{lon:.6f}_{int(time.time())}"

        # Vary pitch for better coverage
        pitch_options = [
            config.PITCH_RANGE[0],
            0,
            config.PITCH_RANGE[1]
        ]

        for heading in range(0, 360, config.HEADING_STEP):
            for pitch in pitch_options:
                params = {
                    "size": f"{config.IMG_WIDTH}x{config.IMG_HEIGHT}",
                    "location": f"{lat},{lon}",
                    "fov": config.FOV,
                    "pitch": pitch,
                    "heading": heading,
                    "key": config.API_KEY,
                    "source": "outdoor",
                    "return_error_code": "true"
                }
                params_list.append((
                    f"{base_filename}_h{heading}_p{pitch}.jpg",
                    params
                ))

        return params_list

    def _download_single_image(self, filename: str, params: dict) -> Optional[Dict]:
        """Download and save a single StreetView image"""
        filepath = config.IMAGE_DIR / filename

        # Skip if file already exists
        if filepath.exists():
            logger.debug(f"Image exists, skipping: {filename}")
            return {
                "path": str(filepath),
                "lat": float(params["location"].split(",")[0]),
                "lon": float(params["location"].split(",")[1]),
                "heading": params["heading"],
                "pitch": params["pitch"],
                "fov": params["fov"],
                "timestamp": datetime.now().isoformat(),
                "status": "exists"
            }

        self._rate_limit()
        response = make_api_request(config.STREETVIEW_URL, params, self.session)

        if not response or not response.content:
            logger.warning(f"Failed to download image: {filename}")
            return None

        if not save_image(response.content, filepath):
            return None

        return {
            "path": str(filepath),
            "lat": float(params["location"].split(",")[0]),
            "lon": float(params["location"].split(",")[1]),
            "heading": params["heading"],
            "pitch": params["pitch"],
            "fov": params["fov"],
            "timestamp": datetime.now().isoformat(),
            "status": "downloaded"
        }

    def download_panorama(self, lat: float, lon: float) -> int:
        """Download complete 360° panorama at a location"""
        downloaded = 0
        params_list = self._generate_pano_params(lat, lon)

        for filename, params in params_list:
            result = self._download_single_image(filename, params)
            if result:
                self.metadata.append(result)
                downloaded += 1

        return downloaded

    def download_area(self, center: Tuple[float, float], radius_meters: float = 500.0) -> Dict[str, int]:
        """Download StreetView images in a defined area"""
        start_time = time.time()
        points = generate_grid_points(center, radius_meters)
        total_downloaded = 0

        logger.info(f"Starting area download with {len(points)} points")

        # Parallel execution
        def process_point(point):
            nonlocal total_downloaded
            lat, lon = point
            count = self.download_panorama(lat, lon)
            total_downloaded += count
            return count

        results = parallel_execute(process_point, points)

        # Save metadata
        if self.metadata:
            save_metadata(self.metadata, config.METADATA_FILE)

        stats = {
            "total_points": len(points),
            "images_downloaded": total_downloaded,
            "total_images": len(self.metadata),
            "elapsed_seconds": time.time() - start_time
        }

        logger.info(f"Download completed. Stats: {stats}")
        return stats

    def get_coverage_stats(self) -> Dict:
        """Calculate coverage statistics"""
        if not self.metadata:
            return {}

        lats = [float(item['lat']) for item in self.metadata]
        lons = [float(item['lon']) for item in self.metadata]

        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)

        center_lat = (min_lat + max_lat) / 2
        center_lon = (min_lon + max_lon) / 2

        # Calculate approximate area covered
        diagonal = calculate_distance((min_lat, min_lon), (max_lat, max_lon))
        area_km2 = math.pi * (diagonal / 2) ** 2 / 1e6

        return {
            "min_lat": min_lat,
            "max_lat": max_lat,
            "min_lon": min_lon,
            "max_lon": max_lon,
            "center": (center_lat, center_lon),
            "diagonal_km": diagonal / 1000,
            "area_km2": area_km2,
            "unique_points": len({(m['lat'], m['lon']) for m in self.metadata}),
            "total_images": len(self.metadata)
        }