from branchingmodel.simulation import simulate_branching_model
import math


if __name__ == "__main__":
    """
    Example usage: run the simulation with default parameters.
    """
    num_neurons_per_layer = 1000
    num_layers = 100
    connections_per_neuron = 4
    transmission_probability = 0.25
    p_spont = 0
    prob_first_layer_on = 0.5
    total_time_steps = num_layers-1

    results = simulate_branching_model(
        num_neurons_per_layer=num_neurons_per_layer,
        num_layers=num_layers,
        connections_per_neuron=connections_per_neuron,
        transmission_probability=transmission_probability,
        p_spont=p_spont,
        prob_first_layer_on=prob_first_layer_on,
        total_time_steps=total_time_steps
    )
    print("Simulation complete.")
    print("Average Branching Ratio:", results['statistics']['avg_branching_ratio'])
    # print("Sample of Avalanche Histogram:", results['statistics']['avalanche_histogram'])
    print("Last-layer neuron states average:", results['raw_results']['simulation_log'][-1]['last_layer_neuron_states_avg'])


    eps = 1e-6
    tol = 1
    p_star = prob_first_layer_on
    while tol>eps:
        p_star_next = 1 - math.exp(-connections_per_neuron*transmission_probability*p_star)
        tol = abs(p_star - p_star_next)
        p_star = p_star_next
    print("p_star:", p_star)