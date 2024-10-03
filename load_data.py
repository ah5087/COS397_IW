import os
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset
import numpy as np

# allow loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

class CellMigrationDataset(Dataset):
    def __init__(self, data_dir):
        super().__init__()
        self.data_dir = data_dir
        self.file_names = sorted([f for f in os.listdir(data_dir) if f.endswith('.tif')])

        # find the largest image dims
        self.max_width, self.max_height = self.find_max_dimensions()

    def find_max_dimensions(self):
        """Find the maximum width and height among all TIFF images."""
        max_width, max_height = 0, 0
        for file_name in self.file_names:
            img = Image.open(os.path.join(self.data_dir, file_name))
            width, height = img.size
            max_width = max(max_width, width)
            max_height = max(max_height, height)
        return max_width, max_height

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        tiff_path = os.path.join(self.data_dir, self.file_names[idx])
        try:
            img = Image.open(tiff_path)
            img = np.array(img)

            # check if TIFF file is 2D or 3D
            if img.ndim == 2:
                initial_cell_density = img
                illumination_pattern = img  # duplicate the single channel
            elif img.ndim == 3:
                initial_cell_density = img[:, :, 0]  # first channel density
                illumination_pattern = img[:, :, 1]  # second channel illumination
            else:
                raise ValueError(f"Unexpected number of dimensions in TIFF file: {img.ndim}")

            # pad image to the maximum width and height -- FIX LATER
            padded_density = np.pad(initial_cell_density, 
                                    ((0, self.max_height - initial_cell_density.shape[0]), 
                                     (0, self.max_width - initial_cell_density.shape[1])),
                                    mode='constant', constant_values=0)
            padded_illumination = np.pad(illumination_pattern, 
                                         ((0, self.max_height - illumination_pattern.shape[0]), 
                                          (0, self.max_width - illumination_pattern.shape[1])),
                                         mode='constant', constant_values=0)

            # stack cell density + illumination pattern
            state = np.stack([padded_density, padded_illumination], axis=0)

            # PLACEHOLDER target movement
            target_movement = np.random.rand(101 * 101)

            return torch.tensor(state, dtype=torch.float32), torch.tensor(target_movement, dtype=torch.float32)

        except OSError as e:
            print(f"Error loading image: {tiff_path}. Error: {e}")
            return None
