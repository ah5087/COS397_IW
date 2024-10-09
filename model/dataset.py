import os
import numpy as np
import torch
from torch.utils.data import Dataset
import tifffile
from PIL import Image

class CellMigrationDataset(Dataset):
    def __init__(self, data_dir, resize_shape=(101, 101)):
        # Recursively collect all .tif files in the directories
        self.data_dir = data_dir
        self.file_paths = [os.path.join(root, f) 
                           for root, _, files in os.walk(data_dir) 
                           for f in files if f.endswith('.tif')]
        self.resize_shape = resize_shape

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        tiff_path = self.file_paths[idx]
        try:
            img = tifffile.imread(tiff_path)  # Load the .TIFF time-lapse

            img = img.astype(np.float32)

            # If the .TIFF is a time-lapse (3D), process the first few frames
            if img.ndim == 3:
                initial_cell_density = img[0]  # Assume first frame is the cell density
                illumination_pattern = img[1]  # Assume second frame is the illumination pattern
            else:
                raise ValueError(f"Unexpected number of dimensions in TIFF file: {img.ndim}")

            # Resize if necessary
            initial_cell_density = np.array(
                Image.fromarray(initial_cell_density).resize(self.resize_shape, Image.BILINEAR)
            )
            illumination_pattern = np.array(
                Image.fromarray(illumination_pattern).resize(self.resize_shape, Image.BILINEAR)
            )

            # Stack the cell density and illumination pattern to form the state
            state = np.stack([initial_cell_density, illumination_pattern], axis=0)

            # Placeholder for the target movement (modify this according to your use case)
            target_movement = np.random.rand(*self.resize_shape)

            return torch.tensor(state, dtype=torch.float32), torch.tensor(target_movement, dtype=torch.float32)

        except Exception as e:
            print(f"Error loading image: {tiff_path}. Error: {e}")
            return torch.zeros((2, *self.resize_shape), dtype=torch.float32), torch.zeros(self.resize_shape, dtype=torch.float32)

