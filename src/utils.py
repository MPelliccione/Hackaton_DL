import torch
import random
import numpy as np
import tarfile
import os

def set_seed(seed=777):
    seed = seed
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

class RandomEdgeDrop:
    def __init__(self, p: float = 0.2):
        assert 0.0 <= p <= 1.0
        self.p = p

    def call(self, data: Data) -> Data:
        edge_index = data.edge_index
        num_edges = edge_index.size(1)

        # Genera maschera booleana: True = edge mantenuto
        mask = torch.rand(num_edges) > self.p
        data.edge_index = edge_index[:, mask]

        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            data.edge_attr = data.edge_attr[mask]

        return data

class GaussianEdgeNoise:
    def __init__(self, std: float = 0.08, p: float = 1.0):
        """
        Applica rumore gaussiano alle edge features.

        Args:
            std: Deviazione standard del rumore gaussiano
            p: Probabilità di applicare il rumore (1.0 = sempre)
        """
        assert std >= 0.0
        assert 0.0 <= p <= 1.0
        self.std = std
        self.p = p

    def __call__(self, data: Data) -> Data:
        if torch.rand(1).item() > self.p:
            return data

        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            # Genera rumore gaussiano con la stessa forma delle edge features
            noise = torch.randn_like(data.edge_attr) * self.std
            # Applica il rumore e assicurati che i valori rimangano nel range [0, 1]
            noisy_features = data.edge_attr + noise
            # Clamp per mantenere i valori nel range delle confidence scores [0, 1]
            data.edge_attr = torch.clamp(noisy_features, 0.0, 1.0)

        return data

class ComposedTransform:
    def __init__(self, transforms):
        """
        Compone multiple trasformazioni in sequenza.

        Args:
            transforms: Lista di trasformazioni da applicare in ordine
        """
        self.transforms = transforms

    def __call__(self, data: Data) -> Data:
        for transform in self.transforms:
            data = transform(data)
        return data

def gzip_folder(folder_path, output_file):
    """
    Compresses an entire folder into a single .tar.gz file.
    
    Args:
        folder_path (str): Path to the folder to compress.
        output_file (str): Path to the output .tar.gz file.
    """
    with tarfile.open(output_file, "w:gz") as tar:
        tar.add(folder_path, arcname=os.path.basename(folder_path))
    print(f"Folder '{folder_path}' has been compressed into '{output_file}'")

# Example usage
# folder_path = "./testfolder/submission"            # Path to the folder you want to compress
# output_file = "./testfolder/submission.gz"         # Output .gz file name
# gzip_folder(folder_path, output_file)