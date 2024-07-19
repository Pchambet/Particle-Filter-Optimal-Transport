#%%

# optimal_transport.py

import numpy as np
import matplotlib.pyplot as plt
import ot

def simulate_state_trajectory(simTime, x0, Q):
    """
    Simulate the true state trajectory of the system.
    
    Parameters:
    - simTime: Total simulation time.
    - x0: Initial state.
    - Q: Process noise variance.
    
    Returns:
    - stateTrajectory: Array of true state values over time.
    """
    stateTrajectory = np.zeros(simTime + 1)
    stateTrajectory[0] = x0

    for i in range(simTime):
        stateTrajectory[i + 1] = (
            0.5 * stateTrajectory[i]
            + 25 * (stateTrajectory[i] / (1 + stateTrajectory[i] ** 2))
            + 8 * np.cos(1.2 * i)
            + np.random.normal(0, Q)
        )
    return stateTrajectory

def generate_observations(stateTrajectory, R):
    """
    Generate observations based on the true state trajectory.
    
    Parameters:
    - stateTrajectory: Array of true state values.
    - R: Observation noise variance.
    
    Returns:
    - observations: Array of noisy observations.
    """
    return (stateTrajectory ** 2) / 20 + np.random.normal(0, R, len(stateTrajectory))

def transition(particles, n, Q):
    """
    Transition function for the particle filter.
    
    Parameters:
    - particles: Array of current particle states.
    - n: Current time step.
    - Q: Process noise variance.
    
    Returns:
    - New particle states after applying the transition model.
    """
    return (
        0.5 * particles
        + 25 * particles / (1 + particles ** 2)
        + 8 * np.cos(1.2 * n)
        + np.random.normal(0, Q, size=particles.shape)
    )

def likelihood_fn(particles, observation, R):
    """
    Likelihood function to update particle weights.
    
    Parameters:
    - particles: Array of particle states.
    - observation: Current observation.
    - R: Observation noise variance.
    
    Returns:
    - Likelihoods for each particle.
    """
    predicted_observation = particles ** 2 / 20
    return np.exp(-0.5 * ((predicted_observation - observation) ** 2) / R)

def resampling(particles, weights):
    """
    Classic resampling method for particle filter.
    
    Parameters:
    - particles: Array of particle states.
    - weights: Array of particle weights.
    
    Returns:
    - Resampled particles.
    """
    indices = np.random.choice(np.arange(len(particles)), size=len(particles), p=weights)
    return particles[indices]

def optimal_transport_resampling(particles, weights, reg=0.01, numItermax=100):
    """
    Optimal Transport resampling method for particle filter.
    
    Parameters:
    - particles: Array of particle states.
    - weights: Array of particle weights.
    - reg: Regularization parameter for the Sinkhorn algorithm.
    - numItermax: Maximum number of iterations for the Sinkhorn algorithm.
    
    Returns:
    - Resampled particles.
    """
    N = len(weights)
    target_weights = np.ones(N) / N
    M = ot.dist(particles.reshape((N, 1)), particles.reshape((N, 1)))
    G = ot.sinkhorn(weights, target_weights, M, reg=reg, numItermax=numItermax)
    indices = np.random.choice(np.arange(N), size=N, p=np.sum(G, axis=1))
    return particles[indices]

def particle_filter(simTime, observations, Q, R, resample_fn):
    """
    Run the particle filter algorithm.
    
    Parameters:
    - simTime: Total simulation time.
    - observations: Array of observations.
    - Q: Process noise variance.
    - R: Observation noise variance.
    - resample_fn: Resampling function to use.
    
    Returns:
    - state_estimates: Array of estimated states over time.
    """
    particles = np.random.randn(500)
    weights = np.ones_like(particles) / len(particles)
    state_estimates = [0]

    for i in range(1, simTime + 1):
        particles = transition(particles, i, Q)
        weights = likelihood_fn(particles, observations[i], R)
        weights += 1e-300  # Avoid division by zero
        weights /= np.sum(weights)
        state_estimate = np.sum(particles * weights)
        particles = resample_fn(particles, weights)
        weights.fill(1.0 / len(particles))
        state_estimates.append(state_estimate)

    return state_estimates

def plot_results(true_states, observations, state_estimates_classic, state_estimates_diff, cmse_classic, cmse_diff):
    """
    Plot the results of the particle filter.
    
    Parameters:
    - true_states: Array of true state values.
    - observations: Array of observations.
    - state_estimates_classic: Array of state estimates using classic resampling.
    - state_estimates_diff: Array of state estimates using optimal transport resampling.
    - cmse_classic: Cumulative Mean Squared Error for classic resampling.
    - cmse_diff: Cumulative Mean Squared Error for optimal transport resampling.
    """
    plt.figure(figsize=(14, 8))
    plt.plot(true_states, label='True States')
    plt.plot(state_estimates_classic, label='Classic Resampling', color='red')
    plt.plot(state_estimates_diff, label='Optimal Transport Resampling', color='black')
    plt.plot(observations, label='Observations', color='blue')
    plt.xlabel("Time")
    plt.ylabel("State")
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(14, 8))
    plt.plot(cmse_classic, label='CMSE Classic', color='red')
    plt.plot(cmse_diff, label='CMSE Optimal Transport', color='black')
    plt.xlabel("Time")
    plt.ylabel("CMSE")
    plt.legend()
    plt.grid()
    plt.show()

def main():
    """
    Main function to run the simulation and particle filter.
    """
    simTime = 30
    x0 = 0
    Q = 10
    R = 1

    # Simulate the true state trajectory
    stateTrajectory = simulate_state_trajectory(simTime, x0, Q)
    
    # Generate noisy observations
    observations = generate_observations(stateTrajectory, R)
    
    # Estimate states using classic and optimal transport resampling methods
    state_estimates_classic = particle_filter(simTime, observations, Q, R, resampling)
    state_estimates_diff = particle_filter(simTime, observations, Q, R, optimal_transport_resampling)
    
    # Calculate true states and errors
    true_states = stateTrajectory[:simTime + 1]
    mse_classic = np.mean((np.array(state_estimates_classic) - true_states) ** 2)
    mse_diff = np.mean((np.array(state_estimates_diff) - true_states) ** 2)
    
    cmse_classic = np.cumsum((np.array(state_estimates_classic) - true_states) ** 2) / np.arange(1, simTime + 2)
    cmse_diff = np.cumsum((np.array(state_estimates_diff) - true_states) ** 2) / np.arange(1, simTime + 2)
    
    print(f"MSE for Classic Resampling: {mse_classic}")
    print(f"MSE for Optimal Transport Resampling: {mse_diff}")
    
    # Plot the results
    plot_results(true_states, observations, state_estimates_classic, state_estimates_diff, cmse_classic, cmse_diff)

if __name__ == "__main__":
    main()
