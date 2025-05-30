import os
import time
import logging
from typing import List, Dict, Optional, Set
from pathlib import Path
from datetime import datetime
import json
import requests
from typing import Optional, Tuple

from .config import config
from .utils import (
    setup_logging,
    ensure_directory_exists,
    save_image,
    save_metadata,
    make_api_request,
    parallel_execute
)

logger = setup_logging(config.LOG_FILE)


class PlacePhotoDownloader:
    def __init__(self, session: Optional[requests.Session] = None):
        ensure_directory_exists(config.IMAGE_DIR)
        self.metadata = []
        self.session = session or requests.Session()
        self.last_request_time = 0
        self.processed_places: Set[str] = set()

    def _rate_limit(self):
        """Enforce rate limiting between API calls"""
        elapsed = time.time() - self.last_request_time
        if elapsed < config.RATE_LIMIT_DELAY:
            time.sleep(config.RATE_LIMIT_DELAY - elapsed)
        self.last_request_time = time.time()

    def get_nearby_places(self, location: Tuple[float, float], radius: int = 100,
                          place_type: Optional[str] = None) -> List[Dict]:
        """Get nearby places with pagination handling"""
        all_results = []
        params = {
            "location": f"{location[0]},{location[1]}",
            "radius": radius,
            "key": config.API_KEY
        }

        if place_type:
            params["type"] = place_type

        while True:
            self._rate_limit()
            response = make_api_request(config.PLACES_URL, params, self.session)

            if not response:
                break

            data = response.json()

            if data.get("status") != "OK":
                logger.warning(f"Places API error: {data.get('status')}")
                break

            all_results.extend(data.get("results", []))

            # Handle pagination
            next_page_token = data.get("next_page_token")
            if not next_page_token:
                break

            params["pagetoken"] = next_page_token
            time.sleep(2)  # Required between page token usage

        return all_results

    def get_place_details(self, place_id: str) -> Optional[Dict]:
        """Get detailed information about a place"""
        params = {
            "place_id": place_id,
            "fields": "name,types,photos,geometry,formatted_address",
            "key": config.API_KEY
        }

        self._rate_limit()
        response = make_api_request(config.PLACE_DETAILS_URL, params, self.session)

        if not response:
            return None

        data = response.json()
        if data.get("status") == "OK":
            return data.get("result")
        return None

    def download_place_photo(self, photo_ref: str, place_id: str, photo_index: int) -> Optional[Dict]:
        """Download a single place photo"""
        filename = f"place_{place_id}_{photo_index}.jpg"
        filepath = config.IMAGE_DIR / filename

        if filepath.exists():
            logger.debug(f"Place photo exists, skipping: {filename}")
            return {
                "path": str(filepath),
                "place_id": place_id,
                "photo_index": photo_index,
                "timestamp": datetime.now().isoformat(),
                "status": "exists"
            }

        params = {
            "maxwidth": config.IMG_WIDTH,
            "photoreference": photo_ref,
            "key": config.API_KEY
        }

        self._rate_limit()
        response = make_api_request(config.PLACE_PHOTOS_URL, params, self.session)

        if not response or not response.content:
            logger.warning(f"Failed to download place photo: {filename}")
            return None

        if not save_image(response.content, filepath):
            return None

        return {
            "path": str(filepath),
            "place_id": place_id,
            "photo_index": photo_index,
            "timestamp": datetime.now().isoformat(),
            "status": "downloaded"
        }

    def process_single_place(self, place: Dict) -> int:
        """Process and download all photos for a single place"""
        place_id = place.get("place_id")
        if not place_id or place_id in self.processed_places:
            return 0

        # Get full place details
        place_details = self.get_place_details(place_id)
        if not place_details:
            return 0

        photos = place_details.get("photos", [])
        if not photos:
            return 0

        downloaded = 0
        place_metadata = {
            "place_id": place_id,
            "name": place_details.get("name"),
            "types": place_details.get("types", []),
            "address": place_details.get("formatted_address"),
            "location": place_details.get("geometry", {}).get("location"),
            "photos": []
        }

        for i, photo in enumerate(photos[:config.MAX_PHOTOS_PER_PLACE]):
            photo_ref = photo.get("photo_reference")
            if not photo_ref:
                continue

            result = self.download_place_photo(photo_ref, place_id, i)
            if result:
                place_metadata["photos"].append(result)
                downloaded += 1

        if downloaded > 0:
            self.metadata.append(place_metadata)
            self.processed_places.add(place_id)

        return downloaded

    def download_places_in_area(self, center: Tuple[float, float], radius: int = 500,
                                place_types: Optional[List[str]] = None) -> Dict[str, int]:
        """Download photos for all places in the specified area"""
        if place_types is None:
            place_types = [
                'cafe', 'restaurant', 'bar',
                'school', 'university', 'park',
                'museum', 'place_of_worship',
                'shopping_mall', 'train_station'
            ]

        start_time = time.time()
        total_downloaded = 0
        stats = {}

        for place_type in place_types:
            logger.info(f"Processing places of type: {place_type}")
            places = self.get_nearby_places(center, radius, place_type)

            if not places:
                continue

            # Process places in parallel
            def process_place_wrapper(place):
                return self.process_single_place(place)

            results = parallel_execute(process_place_wrapper, places)
            type_count = sum(results)

            stats[place_type] = {
                "places_found": len(places),
                "photos_downloaded": type_count
            }
            total_downloaded += type_count

        # Save metadata
        if self.metadata:
            save_metadata(self.metadata, config.METADATA_FILE)

        stats["total_photos_downloaded"] = total_downloaded
        stats["elapsed_seconds"] = time.time() - start_time

        logger.info(f"Place photo download completed. Stats: {stats}")
        return stats