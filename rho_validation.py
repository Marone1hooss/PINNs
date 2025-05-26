import torch
import numpy as np
import matplotlib.pyplot as plt

def validate_outlet_density(model, pipe_length, t_values, inlet_density, speed, device):
    """
    Validates the outlet density by comparing the PINN prediction with the theoretical value
    based on inlet density and advection speed.
    
    Args:
        model: The trained PINN model
        pipe_length: Length of the pipe
        t_values: Time points to evaluate
        inlet_density: Initial/inlet density values
        speed: Advection speed
        device: torch device to use
    
    Returns:
        tuple: (predicted_density, theoretical_density, error)
    """
    # Create outlet position tensor
    x_outlet = torch.full((len(t_values), 1), pipe_length, device=device)
    t_tensor = torch.tensor(t_values, device=device).view(-1, 1)
    
    # Get PINN prediction at outlet
    with torch.no_grad():
        predicted_density = model(x_outlet, t_tensor).cpu().numpy().flatten()
    
    # Calculate theoretical density at outlet
    # For a given time t, the density at outlet should be the inlet density at time (t - L/a)
    theoretical_times = t_values - pipe_length/speed
    theoretical_times = np.maximum(theoretical_times, 0)  # Ensure non-negative times
    theoretical_density = inlet_density(theoretical_times)
    
    # Calculate error
    error = np.abs(predicted_density - theoretical_density)
    
    return predicted_density, theoretical_density, error

def validate_final_time(model, pipe_length, x_points, t_final, device):
    """
    Validates the density prediction at the final time T_final.
    
    Args:
        model: The trained PINN model
        pipe_length: Length of the pipe
        x_points: Spatial points to evaluate
        t_final: Final time point
        device: torch device to use
    
    Returns:
        tuple: (x_points, predicted_density)
    """
    # Create tensors for evaluation
    x_tensor = torch.tensor(x_points, device=device).view(-1, 1)
    t_tensor = torch.full_like(x_tensor, t_final)
    
    # Get PINN prediction at final time
    with torch.no_grad():
        predicted_density = model(x_tensor, t_tensor).cpu().numpy().flatten()
    
    return x_points, predicted_density

def plot_validation_results(t_eval, pred_dens, theo_dens, x_points, final_density, t_final):
    """
    Plots the validation results for both outlet density and final time.
    
    Args:
        t_eval: Time points for outlet validation
        pred_dens: Predicted density at outlet
        theo_dens: Theoretical density at outlet
        x_points: Spatial points for final time validation
        final_density: Density at final time
        t_final: Final time value
    """
    # Plot outlet validation results
    plt.figure(figsize=(10, 6))
    plt.plot(t_eval, pred_dens, 'b-', label='PINN Prediction')
    plt.plot(t_eval, theo_dens, 'r--', label='Theoretical')
    plt.xlabel('Time (s)')
    plt.ylabel('Density')
    plt.title('Outlet Density Validation')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Plot final time validation
    plt.figure(figsize=(10, 6))
    plt.plot(x_points, final_density, 'b-')
    plt.xlabel('Position (m)')
    plt.ylabel('Density')
    plt.title(f'Density at T = {t_final}s')
    plt.grid(True)
    plt.show() 