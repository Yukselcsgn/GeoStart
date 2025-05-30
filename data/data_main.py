#data/data_main
"""
Main execution script for StreetView Image Downloader
"""

import argparse
import json
import time
from pathlib import Path
from typing import Tuple

from data import (
    StreetViewDownloader,
    PlacePhotoDownloader,
    RoadNetworkDownloader,
    config
)
from data.utils import save_metadata


def parse_coordinates(coord_str: str) -> Tuple[float, float]:
    """Parse latitude,longitude string into tuple"""
    try:
        lat, lon = map(float, coord_str.split(','))
        return lat, lon
    except ValueError:
        raise argparse.ArgumentTypeError("Coordinates must be 'lat,lon'")

    logging.info("Street View API isteği gönderiliyor.")
def main():
    parser = argparse.ArgumentParser(
        description="StreetView Image Downloader",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'coordinates',
        type=parse_coordinates,
        nargs='?',
        default="39.990139, 32.691874",  # Ankara Cakırlar koordinatları örnek
        help="Center coordinates as 'latitude,longitude'"
    )
    parser.add_argument(
        '-r', '--radius',
        type=float,
        default=0.1,
        help="Radius in kilometers"
    )
    parser.add_argument(
        '--no-streetview',
        action='store_true',
        help="Skip StreetView image download"
    )
    parser.add_argument(
        '--no-places',
        action='store_true',
        help="Skip place photo download"
    )
    parser.add_argument(
        '--road-sampling-interval',
        type=float,
        default=50.0,
        help="Distance between road sampling points in meters"
    )
    parser.add_argument(
        '--max-photos-per-place',
        type=int,
        default=5,
        help="Maximum photos to download per place"
    )

    args = parser.parse_args()

    # Update config with command-line arguments
    config.ROAD_SAMPLING_INTERVAL = args.road_sampling_interval
    config.MAX_PHOTOS_PER_PLACE = args.max_photos_per_place

    # Initialize downloaders
    road_downloader = RoadNetworkDownloader()
    sv_downloader = StreetViewDownloader()
    place_downloader = PlacePhotoDownloader()

    # Execute downloads
    start_time = time.time()

    if not args.no_streetview:
        print("\n=== Downloading StreetView Images ===")
        road_points = road_downloader.get_road_network(args.coordinates, args.radius)
        print(f"Found {len(road_points)} road sampling points")

        sv_stats = sv_downloader.download_area(args.coordinates, args.radius * 1000)
        print("\nStreetView Download Stats:")
        print(json.dumps(sv_stats, indent=2))

    if not args.no_places:
        print("\n=== Downloading Place Photos ===")
        place_stats = place_downloader.download_places_in_area(
            args.coordinates,
            int(args.radius * 1000)
        )
        print("\nPlace Photo Download Stats:")
        print(json.dumps(place_stats, indent=2))

    # Save combined metadata
    combined_metadata = {
        "coordinates": args.coordinates,
        "radius_km": args.radius,
        "streetview": sv_downloader.get_coverage_stats() if not args.no_streetview else None,
        "places": place_stats if not args.no_places else None,
        "execution_time_seconds": time.time() - start_time
    }

    metadata_path = Path("download_summary.json")
    save_metadata(combined_metadata, metadata_path)
    print(f"\nDownload summary saved to {metadata_path}")


if __name__ == "__main__":
    main()