from collections.abc import Callable
import numpy as np

def monte_carlo_simulation(num_simulations: int, num_steps: int, initial_state, step_function: Callable):
    np.random.seed(42)

    simulation_results = [] # Stores each path simulation

    for _ in range(num_simulations):
        simulation_path = [initial_state] # Stores each step for one path simulation

        for step in range(num_steps):
            state = step_function(simulation_path[-1], step)
            simulation_path.append(state)

        simulation_results.append(simulation_path)

    return np.array(simulation_results)