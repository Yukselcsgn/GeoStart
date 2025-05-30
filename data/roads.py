import os
import time
import logging
import math
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import json
import requests
import overpy  # For OpenStreetMap data

from .config import config
from .utils import (
    setup_logging,
    ensure_directory_exists,
    save_metadata,
    calculate_distance,
    parallel_execute
)

logger = setup_logging(config.LOG_FILE)


class RoadNetworkDownloader:
    def __init__(self):
        ensure_directory_exists(config.IMAGE_DIR)
        self.road_network = []
        self.api = overpy.Overpass()

    def _query_osm_roads(self, bbox: Tuple[float, float, float, float]) -> Optional[overpy.Result]:
        """Query OpenStreetMap for road data within bounding box"""
        query = f"""
        [out:json];
        (
            way["highway"~"motorway|trunk|primary|secondary|tertiary|residential|service"]["highway"!~"path|footway|cycleway|pedestrian|steps|track"]{bbox};
        );
        out body;
        >;
        out skel qt;
        """

        try:
            return self.api.query(query)
        except Exception as e:
            logger.error(f"OSM query failed: {str(e)}")
            return None

    def _sample_points_along_road(self, nodes: List[overpy.Node]) -> List[Tuple[float, float]]:
        """Generate sampling points along a road segment"""
        if len(nodes) < 2:
            return []

        points = []
        total_length = 0
        segments = []

        # Calculate segment lengths
        for i in range(len(nodes) - 1):
            n1 = nodes[i]
            n2 = nodes[i + 1]
            dist = calculate_distance((n1.lat, n1.lon), (n2.lat, n2.lon))
            segments.append((n1, n2, dist))
            total_length += dist

        if total_length == 0:
            return []

        # Calculate number of sampling points
        num_points = max(2, math.ceil(total_length / config.ROAD_SAMPLING_INTERVAL))

        # Generate evenly spaced points
        for i in range(num_points):
            target_dist = (i / (num_points - 1)) * total_length
            accumulated = 0

            for seg in segments:
                n1, n2, dist = seg
                if accumulated + dist >= target_dist or seg == segments[-1]:
                    # Linear interpolation
                    ratio = (target_dist - accumulated) / dist if dist > 0 else 0
                    lat = float(n1.lat) + ratio * (float(n2.lat) - float(n1.lat))
                    lon = float(n1.lon) + ratio * (float(n2.lon) - float(n1.lon))
                    points.append((lat, lon))
                    break

                accumulated += dist

        return points

    def get_road_network(self, center: Tuple[float, float], radius_km: float = 1.0) -> List[Tuple[float, float]]:
        """Get complete road network within radius of center"""
        # Convert radius to bounding box
        lat, lon = center
        radius_deg = radius_km / 111.32  # Approximate conversion

        bbox = (
            lat - radius_deg,
            lon - radius_deg / math.cos(math.radians(lat)),
            lat + radius_deg,
            lon + radius_deg / math.cos(math.radians(lat))
        )

        result = self._query_osm_roads(bbox)
        if not result:
            return []

        road_points = []

        # Process each road way
        for way in result.ways:
            nodes = way.nodes
            if len(nodes) < 2:
                continue

            # Sample points along this road
            sampled_points = self._sample_points_along_road(nodes)
            road_points.extend(sampled_points)

            # Store road metadata
            self.road_network.append({
                "id": way.id,
                "name": way.tags.get("name", ""),
                "type": way.tags.get("highway", ""),
                "nodes": [(n.lat, n.lon) for n in nodes],
                "sampled_points": sampled_points
            })

        # Save road network data
        save_metadata(self.road_network, config.ROAD_NETWORK_FILE)

        return road_points

    def save_road_network_geojson(self, file_path: Path) -> bool:
        """Save road network in GeoJSON format"""
        if not self.road_network:
            return False

        features = []

        for road in self.road_network:
            feature = {
                "type": "Feature",
                "properties": {
                    "id": road["id"],
                    "name": road["name"],
                    "type": road["type"]
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[n[1], n[0]] for n in road["nodes"]]  # GeoJSON uses lon,lat
                }
            }
            features.append(feature)

        geojson = {
            "type": "FeatureCollection",
            "features": features
        }

        return save_metadata(geojson, file_path)

    def visualize_road_network(self):
        """Generate visualization of the road network (placeholder)"""
        # This would typically use matplotlib or folium to create a visualization
        logger.info("Road network visualization would be generated here")
        return True