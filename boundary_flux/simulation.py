import numpy as np
from scipy.integrate import solve_ivp
from model import construct_2D_model_flux_rescaled
from utils import load_and_process_pattern, interpolate_results, plot_results

def run_simulation():
    # Load and process the illumination pattern
    pattern = load_and_process_pattern('illuminati_pattern.mat')

    # Set simulation parameters
    N = 101
    L = 5400  # simulation width in um (5.4 mm = 5,400 um)
    tF = 2880  # simulation time in min (48 h = 2,880 min)

    # Construct the model
    model = construct_2D_model_flux_rescaled(N, L, pattern)

    # Set initial conditions (uniform distribution)
    ic = np.ones(N*N)

    # Solve ODE
    sol = solve_ivp(model, [0, tF], ic, method='BDF')

    # Interpolate results
    tv, yv = interpolate_results(sol.t, sol.y)

    # Save results
    np.savez('Fig6_simulation_results.npz', L=L, N=N, tF=tF, tv=tv, yv=yv)

    # Plot final state
    X, Y = np.meshgrid(np.linspace(0, L, N), np.linspace(0, L, N))
    Z = yv[-1, :].reshape((N, N))
    plot_results(X, Y, Z, 'Boundary-flux model at 48 h', 'fig6c_simulation.png')

if __name__ == "__main__":
    run_simulation()