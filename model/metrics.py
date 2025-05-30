import numpy as np
import torch
from geopy.distance import geodesic


def geodesic_distance_km(coord1, coord2):
    try:
        return geodesic(coord1, coord2).km
    except Exception as e:
        print(f"[Geodesic] Hata: {e} | coord1: {coord1}, coord2: {coord2}")
        return float("inf")


def mean_geodesic_distance(predictions, targets):
    errors = []
    for pred, target in zip(predictions, targets):
        err = geodesic_distance_km(tuple(pred), tuple(target))
        if np.isfinite(err):
            errors.append(err)
    return np.mean(errors) if errors else float("inf")


def compute_haversine_distance(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Torch tensörleri ile haversine (km) mesafesi döndürür.
    """
    lat1, lon1 = preds[:, 0], preds[:, 1]
    lat2, lon2 = targets[:, 0], targets[:, 1]

    lat1 = torch.deg2rad(lat1)
    lon1 = torch.deg2rad(lon1)
    lat2 = torch.deg2rad(lat2)
    lon2 = torch.deg2rad(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))

    R = 6371.0  # km
    distance = R * c
    return distance
