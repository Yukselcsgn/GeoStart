import torch
from tqdm import tqdm
from model.utils import load_image
from model.metrics import geodesic_distance_km

class SelfLearner:
    def __init__(self, model, transform, threshold=0.9, max_error_km=5.0, device=None, verbose=True):
        self.model = model.to(device)
        self.transform = transform
        self.threshold = threshold
        self.max_error_km = max_error_km
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose

    def generate_pseudo_labels(self, unlabeled_df, save_confidence=False):
        """
        Pseudo-labeling: Modelin yüksek güvenli tahminlerine göre etiketli veri üretir.
        `unlabeled_df` → DataFrame with image path and (optional) ground-truth.
        """
        self.model.eval()
        pseudo_data = []

        for idx, row in tqdm(unlabeled_df.iterrows(), total=len(unlabeled_df), desc="Generating Pseudo Labels"):
            img_tensor = load_image(row['path'], self.transform, self.device)
            if img_tensor is None:
                continue

            with torch.no_grad():
                output = self.model(img_tensor.unsqueeze(0))  # shape: [1, 2]
                coords = output.squeeze().cpu().numpy()       # [lat, lon]
                pred_lat, pred_lon = coords[0], coords[1]

            error_km = None
            if 'lat' in row and 'lon' in row:
                gt_coords = (row['lat'], row['lon'])
                pred_coords = (pred_lat, pred_lon)
                error_km = geodesic_distance_km(gt_coords, pred_coords)
                if error_km > self.max_error_km:
                    if self.verbose:
                        print(f"[Skip] Error too high ({error_km:.2f} km) for {row['path']}")
                    continue

            pseudo_entry = {
                "path": row['path'],
                "lat": pred_lat,
                "lon": pred_lon,
                "heading": row.get("heading", 0),
            }

            if save_confidence and error_km is not None:
                pseudo_entry["error_km"] = round(error_km, 2)

            pseudo_data.append(pseudo_entry)

        if self.verbose:
            print(f"[Info] Toplam {len(pseudo_data)} pseudo label üretildi.")
        return pseudo_data
