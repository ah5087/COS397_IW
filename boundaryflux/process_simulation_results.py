import numpy as np
import matplotlib.pyplot as plt
import os

def load_and_process_fig2g():
    # Load the Fig2G simulation results
    data = np.load('Fig2G_simulation_results.npz')
    
    L = data['L']
    N = data['N']
    tF = data['tF']
    tv = data['tv']
    yv = data['yv']
    
    # Process and plot the results
    xv = np.linspace(0, L, N)
    X, Y = np.meshgrid(xv, xv)
    i20 = np.argmax(tv >= 20*60)
    Z20 = yv[i20].reshape(N, N)
    i32 = np.argmax(tv >= 32*60)
    Z32 = yv[i32].reshape(N, N)
    xx = X[50, 50:] - L/2

    plt.figure(figsize=(10, 6))
    plt.plot(xx, Z20[50, 50:], label='20 hours')
    plt.plot(xx, Z32[50, 50:], label='32 hours')
    plt.xlabel('Distance from center (μm)')
    plt.ylabel('Relative cell density')
    plt.ylim(0, 2)
    plt.xlim(0, 1000)
    plt.legend()
    plt.title("Fig2G Results: Cell Density Profile")
    plt.savefig('Fig2G_density_profile.png')
    plt.close()

    # Additional analysis: calculate and plot total cell count over time
    total_cells = np.sum(yv.reshape(-1, N, N), axis=(1, 2))
    plt.figure(figsize=(10, 6))
    plt.plot(tv / 60, total_cells)
    plt.xlabel('Time (hours)')
    plt.ylabel('Total cell count')
    plt.title('Fig2G: Total Cell Count Over Time')
    plt.savefig('Fig2G_total_cell_count.png')
    plt.close()

def load_and_process_fig6():
    # Load the Fig6 simulation results
    data = np.load('Fig6_simulation_results.npz')
    
    L = data['L']
    N = data['N']
    tF = data['tF']
    tv = data['tv']
    yv = data['yv']
    
    # Process and plot the results
    xv = np.linspace(0, L, N)
    X, Y = np.meshgrid(xv, xv)

    # Plot final state
    final_state = yv[-1].reshape(N, N)
    plt.figure(figsize=(10, 8))
    plt.pcolormesh(X, Y, final_state, vmin=0, vmax=1.2)
    plt.colorbar(label='Cell density')
    plt.title(f'Fig6: Final Cell Distribution at t = {tv[-1]/60:.1f} hours')
    plt.xlabel('X position (μm)')
    plt.ylabel('Y position (μm)')
    plt.savefig('Fig6_final_distribution.png')
    plt.close()

    # Create an animation (save frames)
    os.makedirs('Fig6_animation_frames', exist_ok=True)
    for i, t in enumerate(tv):
        if i % 10 == 0:  # Save every 10th frame to reduce file count
            plt.figure(figsize=(10, 8))
            plt.pcolormesh(X, Y, yv[i].reshape(N, N), vmin=0, vmax=1.2)
            plt.colorbar(label='Cell density')
            plt.title(f'Fig6: Cell Distribution at t = {t/60:.1f} hours')
            plt.xlabel('X position (μm)')
            plt.ylabel('Y position (μm)')
            plt.savefig(f'Fig6_animation_frames/frame_{i:03d}.png')
            plt.close()

if __name__ == "__main__":
    load_and_process_fig2g()
    load_and_process_fig6()
    print("Processing complete. Check the output files.")