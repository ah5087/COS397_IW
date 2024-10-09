import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.interpolate import interp1d

def load_and_process_pattern(file_path):
    pattern_data = loadmat(file_path)
    pattern = pattern_data['pattern']
    pattern = np.flipud(pattern)
    pattern[:, :2] = 0
    pattern[:, -2:] = 0
    pattern[:2, :] = 0
    pattern[-2:, :] = 0
    pattern = pattern > 0
    return pattern

def interpolate_results(t, y, num_points=290):
    tv = np.linspace(0, t[-1], num_points)
    
    # Ensure y is 2D
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    
    # Transpose y if necessary
    if y.shape[0] != len(t):
        y = y.T
    
    yv = np.array([np.interp(tv, t, y[:, i]) for i in range(y.shape[1])]).T
    return tv, yv

def plot_results(X, Y, Z, title, output_file):
    plt.figure(figsize=(10, 8))
    plt.pcolormesh(X, Y, Z, cmap='viridis', vmin=0, vmax=1.2)
    plt.colorbar(label='Relative cell density')
    plt.title(title)
    plt.xlabel('μm')
    plt.ylabel('μm')
    plt.axis('square')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()