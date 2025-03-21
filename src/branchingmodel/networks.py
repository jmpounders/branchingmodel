from typing import Dict, List
from collections import defaultdict
from itertools import accumulate
import random

def initialize_feedforward_network(
        num_neurons_per_layer: int,
        connections_per_neuron: int,
        num_layers: int
    ) -> Dict[int, List[int]]:
    """
    Initializes the network as a feedforward network organized into layers.

    Creates a feedforward network with a fixed number of neurons in each layer. Each neuron in layer ℓ can only connect
    to neurons in layer ℓ+1. The total number of neurons in the network will be num_neurons_per_layer * num_layers.

    Args:
        num_neurons_per_layer (int): Number of neurons in each layer.
        connections_per_neuron (int): Fixed number of outgoing connections each neuron can make.
        num_layers (int): Number of layers in the network.

    Returns:
        Dict[int, List[int]]: An adjacency list representing the feedforward network structure.
    """

    # Create layers as lists of neuron indices
    layers = [
        list(range(i * num_neurons_per_layer, (i + 1) * num_neurons_per_layer))
        for i in range(num_layers)
    ]

    adjacency_list = defaultdict(list)

    # Create feedforward connections only from layer l to layer l+1.
    for layer in range(len(layers) - 1):
        next_layer = layers[layer + 1]
        for neuron in layers[layer]:
            # Determine number of connections (capped by available neurons in the next layer)
            num_available = len(next_layer)
            n_conn = min(connections_per_neuron, num_available)
            # Randomly sample n_conn distinct targets from next_layer
            targets = random.sample(next_layer, n_conn)
            adjacency_list[neuron].extend(targets)

    # Neurons in the final layer have no outgoing connections.
    for neuron in layers[-1]:
        adjacency_list[neuron] = []

    return dict(adjacency_list)