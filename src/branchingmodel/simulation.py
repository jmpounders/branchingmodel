import random
from collections import defaultdict
from typing import Dict, Any, List

from branchingmodel.networks import initialize_feedforward_network


def simulation_step(
    neuron_states: List[int],
    ancestors: List[Any],
    avalanches: Dict[str, int],
    adjacency_list: Dict[int, List[int]],
    transmission_probability: float,
    p_spont: float,
    t: int,
) -> tuple[List[int], List[Any], Dict[str, int], List[int], float]:
    """
    Performs a single step of the branching model simulation.

    Args:
        neuron_states (List[int]): Current state of each neuron (0 = off, 1 = on)
        ancestors (List[Any]): List tracking ancestor neurons for avalanche tracking
        avalanches (Dict[str, int]): Dictionary tracking avalanche sizes
        adjacency_list (Dict[int, List[int]]): Network connectivity
        transmission_probability (float): Probability of transmission for each active connection
        p_spont (float): Spontaneous firing probability
        t (int): Current time step

    Returns:
        tuple containing:
        - next_states (List[int]): Updated neuron states
        - ancestors (List[Any]): Updated ancestor tracking
        - avalanches (Dict[str, int]): Updated avalanche sizes
        - currently_active (List[int]): List of currently active neurons
        - branching_ratio (float): Branching ratio for this step
    """
    num_neurons = len(neuron_states)

    # Step 1: Some neurons fire spontaneously
    for i in range(num_neurons):
        if random.random() < p_spont:
            neuron_states[i] = 1
            ancestors[i] = None
            avalanches[f'{i}_{t}'] = 1

    # Step 2: Track active neurons
    currently_active = [i for i, state in enumerate(neuron_states) if state == 1]

    # Step 3: Prepare next-step states
    next_states = [0] * num_neurons
    for active_neuron in currently_active:
        targets = adjacency_list.get(active_neuron, [])
        for target in targets:
            if random.random() < transmission_probability:
                next_states[target] = 1
                ancestors[target] = f'{active_neuron}_{t}' if ancestors[active_neuron] is None else ancestors[active_neuron]
                avalanches[ancestors[target]] += 1

    active_next = [i for i, state in enumerate(next_states) if state == 1]
    for active_neuron in currently_active:
        if active_neuron not in active_next:
            ancestors[active_neuron] = None

    # Calculate branching ratio
    branching_ratio = sum(next_states) / len(currently_active) if currently_active else 0.0

    return next_states, ancestors, avalanches, currently_active, branching_ratio


def run_simulation(
    num_neurons_per_layer: int,
    num_layers: int,
    adjacency_list: Dict[int, List[int]],
    total_time_steps: int,
    transmission_probability: float,
    p_spont: float,
    initial_conditions: List[int],
) -> Dict[str, Any]:
    """
    Runs the branching model simulation over the specified number of time steps.
    Optionally, visualize the network state on each iteration.

    Args:
        num_neurons_per_layer (int): Number of neurons per layer.
        adjacency_list (dict): Mapping from each neuron to the neurons it connects to.
        total_time_steps (int): Number of time steps to simulate.
        transmission_probability (float): Probability of transmission for each active connection.
        p_spont (float): Spontaneous firing probability for each neuron at each time step.

    Returns:
        results (dict): Dictionary with collected data such as avalanche sizes,
                        branching ratio over time, and logs.
    """
    # Active state of each neuron; 0 = off, 1 = on
    neuron_states = [0] * num_neurons_per_layer * num_layers
    ancestors = [None] * num_neurons_per_layer * num_layers
    avalanches = {}

    branching_ratio_series = []
    simulation_log = []

    for neuron in initial_conditions:
        neuron_states[neuron] = 1
        ancestors[neuron] = None
        avalanches[f'{neuron}_{0}'] = 1


    for t in range(total_time_steps):
        # Perform single simulation step
        neuron_states, ancestors, avalanches, currently_active, branching_ratio = simulation_step(
            neuron_states=neuron_states,
            ancestors=ancestors,
            avalanches=avalanches,
            adjacency_list=adjacency_list,
            transmission_probability=transmission_probability,
            p_spont=p_spont,
            t=t
        )
        # for ell in range(num_layers):
        #     print(neuron_states[num_neurons_per_layer * ell:num_neurons_per_layer * (ell + 1)])

        # Average last-layer neutron states
        last_layer_neuron_states = [neuron_states[n] for n in range(num_neurons_per_layer * (num_layers - 1), num_neurons_per_layer * num_layers)]
        last_layer_neuron_states_avg = sum(last_layer_neuron_states) / num_neurons_per_layer
        # print()

        # Log results
        simulation_log.append({
            'time_step': t,
            'active_neurons': currently_active,
            'last_layer_neuron_states_avg': last_layer_neuron_states_avg,
        })

        if currently_active:
            branching_ratio_series.append(branching_ratio)

    results = {
        'avalanches': dict(avalanches),
        'branching_ratio_series': branching_ratio_series,
        'simulation_log': simulation_log
    }
    return results


def gather_statistics(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Gathers basic statistics on avalanche sizes, branching ratio, etc.

    Args:
        results (dict): Simulation outputs containing avalanche sizes and branching ratio data.

    Returns:
        stats (dict): Dictionary with computed statistics (e.g., histogram of avalanche sizes).
    """
    avalanches = results['avalanches']
    branching_ratio_series = results['branching_ratio_series']

    # Simple histogram of avalanche sizes
    avalanche_histogram = defaultdict(int)
    for avalanche_size in avalanches.values():
        avalanche_histogram[avalanche_size] += 1
    avalanche_histogram = dict(sorted(avalanche_histogram.items()))

    # Example average branching ratio
    avg_branching_ratio = sum(branching_ratio_series) / len(branching_ratio_series) if branching_ratio_series else 0

    return {
        'avalanche_histogram': dict(avalanche_histogram),
        'avg_branching_ratio': avg_branching_ratio,
    }


def simulate_branching_model(
    num_neurons_per_layer=10,
    connections_per_neuron=4,
    transmission_probability=0.2,
    p_spont=1e-5,
    prob_first_layer_on=0.5,
    total_time_steps=2,
    num_layers=3,
) -> Dict[str, Any]:
    """
    Main wrapper function to run the Branching Model simulation using a feedforward network topology.

    In this feedforward model, the network is partitioned into multiple layers (by default 10) such that
    neurons in layer ℓ can only connect to neurons in layer ℓ+1. This layered structure emulates a unidirectional,
    hierarchical flow akin to biological feedforward networks.

    Args:
        num_neurons (int): Total number of neurons.
        connections_per_neuron (int): Number of outgoing connections per neuron.
        transmission_probability (float): Probability of transmission on a connection.
        p_spont (float): Spontaneous firing probability for each neuron at each time step.
        prob_first_layer_on (float): Probability of a neuron being on in the first layer.
        total_time_steps (int): Number of simulation time steps.
        num_layers (int): Number of layers for the feedforward network.

    Returns:
        Dict[str, Any]: A dictionary containing simulation logs, avalanche sizes, branching ratio series, and computed statistics.
    """
    # Initialize feedforward network using the layered structure.
    adjacency_list = initialize_feedforward_network(
        num_neurons_per_layer=num_neurons_per_layer,
        connections_per_neuron=connections_per_neuron,
        num_layers=num_layers
    )

    initial_conditions = [n for n in range(num_neurons_per_layer) if random.random() < prob_first_layer_on]

    # Run simulation
    raw_results = run_simulation(
        num_neurons_per_layer=num_neurons_per_layer,
        num_layers=num_layers,
        adjacency_list=adjacency_list,
        total_time_steps=total_time_steps,
        transmission_probability=transmission_probability,
        p_spont=p_spont,
        initial_conditions=initial_conditions,
    )

    # Gather statistics
    stats = gather_statistics(raw_results)

    # Combine all results
    result_bundle = {
        'raw_results': raw_results,
        'statistics': stats
    }
    return result_bundle


