import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt
import os

# Parameters
N = 1000  # Number of agents
L = 4000  # Number of total links
P = 50  # Fixed asset price
p = 0.5  # Probability of fundamental value
q = 0.65  # Deciding probability of private signal
eta_min = 0.7  # Minimum relative influence
eta_max = 3.0  # Maximum relative influence
network_type = "cluster"  # Network type: 'cluster', 'random', 'scale_free'

# Initialize network
if network_type == "scale_free":
    h = 2
    G = nx.barabasi_albert_graph(N, h)
elif network_type == "random":
    network_density = 0.004
    G_p = 2 * network_density / (N - 1)
    G = nx.erdos_renyi_graph(N, G_p)
    # assert network_density == nx.density(G)
elif network_type == "cluster":
    avg_nodes = 4  # Desired average degree

    # Calculate m based on the average degree
    m = int(avg_nodes / 2)  # Convert to integer since m must be an integer
    p = 0.1  # Probability of forming triangles (higher p = higher clustering)

    # Create the power-law cluster graph
    G = nx.powerlaw_cluster_graph(N, m, p)

# Calculate degree centrality
degree_centrality = nx.degree_centrality(G)
relative_influence = {
    node: eta_min + (eta_max - eta_min) * (centrality / max(degree_centrality.values()))
    for node, centrality in degree_centrality.items()
}

def simulate_cascades_with_edge_activation(theta, tau, edge_activation_prob, time_steps=10):
    """
    Simulate decision cascades with edge activation over multiple time steps.
    
    Parameters:
    - theta: Trading fee (affects utility calculation)
    - tau: Search cost (affects whether neighbors are considered)
    - edge_activation_prob: Probability that an edge is active at a given time step
    - time_steps: Number of time steps for the simulation
    
    Returns:
    - cascade_count: Number of agents who changed their decision from their initial state
    - cascade_size: Magnitude of decision imbalances across the network
    - trend_shift_cascade: Overall trend shift as defined in the paper
    """
    # Initialize agent properties
    agents = {
        i: {
            "private_signal": 100 if random.random() < q else 0,
            "decision": "hold",  # Current decision
            "initial_decision": "hold",  # Updated at each time step
            "connected_decisions": [],
        }
        for i in range(N)
    }

    cascade_count = 0
    cascade_size =  0
    trend_shift_cascade = 0
    # Simulate multiple time steps
    for t in range(time_steps):
        for node in sorted(relative_influence, key=relative_influence.get, reverse=True):
            agent = agents[node]

            # Set the initial decision before considering external sources
            agent["initial_decision"] = agent["decision"]

            # Access connected agents' decisions, considering edge activation
            connected_nodes = list(G.neighbors(node))
            observed_decisions = [
                agents[n]["decision"]
                for n in connected_nodes
                if random.random() < edge_activation_prob and random.random() > tau  # Edge activation + search cost
            ]
            agent["connected_decisions"] = observed_decisions

            # Calculate probabilities for decisions
            buy_count = observed_decisions.count("buying")
            sell_count = observed_decisions.count("selling")
            total_influence = sum(
                relative_influence[n] for n in connected_nodes if agents[n]["decision"]
            )
            if total_influence == 0:  # No connected decisions observed
                total_influence = 1  # Avoid division by zero

            # Update decision based on utility
            fv = 100 if agent["private_signal"] == 100 else 0
            def utility(fv, decision):
                if decision == "buying":
                    return fv - P - theta * P
                elif decision == "selling":
                    return P - fv - theta * P
                else:  # holding
                    return 0

            # Determine agent's decision at this time step
            new_decision = "hold"
            if utility(fv, "buying") > max(utility(fv, "selling"), utility(fv, "hold")):
                new_decision = "buying"
            elif utility(fv, "selling") > max(utility(fv, "buying"), utility(fv, "hold")):
                new_decision = "selling"

            # Update the agent's decision
            agent["decision"] = new_decision

        # Analyze results
        # 1. Cascade Count: Count agents who changed their decision from the initial state
        cascade_count += sum(
            1 for agent in agents.values() if agent["initial_decision"] != agent["decision"] and agent["decision"] != "hold"
        )

        # 2. Cascade Size: Magnitude of decision imbalances across the network
        cascade_size += sum(
            abs(agent["connected_decisions"].count("buying") - agent["connected_decisions"].count("selling"))
            for agent in agents.values()
        )

        # # 3. Trend Shift Cascade: Based on the paper's equation
        # initial_buying = sum(1 for agent in agents.values() if agent["initial_decision"] == "buying")
        # final_buying = sum(1 for agent in agents.values() if agent["decision"] == "buying")

        # initial_selling = sum(1 for agent in agents.values() if agent["initial_decision"] == "selling")
        # final_selling = sum(1 for agent in agents.values() if agent["decision"] == "selling")

        # final_holding = sum(1 for agent in agents.values() if agent["decision"] == "hold")

        # trend_shift_cascade += (initial_buying - final_buying) + (initial_selling - final_selling)

    return cascade_count, cascade_size #, trend_shift_cascade


# Iterate through all combinations of tau and theta
tau_values = np.linspace(0, 1, 30)  # 10 values from 0 to 1
theta_values = np.linspace(0, 0.4, 30)  # 10 values from 0 to 0.4
edge_activation_probs = [0, 0.2, 0.4, 0.6, 0.8, 1.0] # Probability that an edge is active at a time step
results_cascade_count = np.zeros((len(edge_activation_probs), len(tau_values), len(theta_values)))
results_cascade_size = np.zeros((len(edge_activation_probs), len(tau_values), len(theta_values)))
for e, prob in enumerate(edge_activation_probs):
    for i, tau in enumerate(tau_values):
        for j, theta in enumerate(theta_values):
            cascade_count, cascade_size = simulate_cascades_with_edge_activation(theta, tau, prob)
            results_cascade_count[e, i, j] = cascade_count
            results_cascade_size[e, i, j] = cascade_size

g_min = np.min(results_cascade_count)
g_max = np.max(results_cascade_count)
t_min = np.min(results_cascade_size)
t_max = np.max(results_cascade_size)

for e, prob in enumerate(edge_activation_probs):
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))

    # Plot results: Number of cascades
    im1 = axs[0].imshow(results_cascade_count[e, :, :], extent=[1, 0, 0, 0.3], vmin=g_min, vmax=g_max, aspect='auto', cmap='viridis')
    axs[0].set_title(f'Number of Cascades vs Tau and Theta with Edge Activation {prob}')
    axs[0].set_ylabel('Theta (Trading Fee)')
    axs[0].set_xlabel('Tau (Search Cost)')
    fig.colorbar(im1, ax=axs[0])

    # Plot results: Cascade size
    im2 = axs[1].imshow(results_cascade_size[e, :, :], extent=[1, 0, 0, 0.3], vmin=t_min, vmax=t_max, aspect='auto', cmap='viridis')
    axs[1].set_title(f'Cascade Size vs Tau and Theta with Edge Activation {prob}')
    axs[1].set_ylabel('Theta (Trading Fee)')
    axs[1].set_xlabel('Tau (Search Cost)')
    fig.colorbar(im2, ax=axs[1])

    plt.tight_layout()
    plt.savefig(f"plot{prob}.png")