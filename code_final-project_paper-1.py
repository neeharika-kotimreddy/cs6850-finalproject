import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt

# Parameters
N = 1000  # Number of agents
L = 4000  # Number of total links
P = 50  # Fixed asset price
p = 0.5  # Probability of fundamental value
q = 0.65  # Deciding probability of private signal
eta_min = 0.7  # Minimum relative influence
eta_max = 3.0  # Maximum relative influence
network_type = "scale_free"  # Network type: 'cluster', 'random', 'scale_free'

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

# Function to simulate cascades
def simulate_cascades(theta, tau):
    # Initialize agent properties
    agents = {
        i: {
            "private_signal": 100 if random.random() < q else 0,
            "decision": "hold",
            "connected_decisions": [],
        }
        for i in range(N)
    }

    # Simulation: Sequential decision-making based on centrality
    for node in sorted(relative_influence, key=relative_influence.get, reverse=True):
        agent = agents[node]
        # Access connected agents' decisions
        connected_nodes = list(G.neighbors(node))
        observed_decisions = [
            agents[n]["decision"]
            for n in connected_nodes
            if random.random() > tau  # Stochastically consider search costs
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

        if utility(fv, "buying") > max(utility(fv, "selling"), utility(fv, "hold")):
            agent["decision"] = "buying"
        elif utility(fv, "selling") > max(utility(fv, "buying"), utility(fv, "hold")):
            agent["decision"] = "selling"
        else:
            agent["decision"] = "hold"

    # Analyze results
    cascade_count = sum(
        1 for agent in agents.values() if agent["decision"] != "hold"
    )
    trend_shift_cascade = sum(
        abs(agent["connected_decisions"].count("buying") - agent["connected_decisions"].count("selling"))
        for agent in agents.values()
    )
    return cascade_count, trend_shift_cascade

# Iterate through all combinations of tau and theta
tau_values = np.linspace(0, 1, 30)  # 10 values from 0 to 1
theta_values = np.linspace(0, 0.3, 30)  # 10 values from 0 to 0.4
results_cascade_count = np.zeros((len(tau_values), len(theta_values)))
results_cascade_size = np.zeros((len(tau_values), len(theta_values)))

for i, tau in enumerate(tau_values):
    for j, theta in enumerate(theta_values):
        cascade_count, cascade_size = simulate_cascades(theta, tau)
        results_cascade_count[i, j] = cascade_count
        results_cascade_size[i, j] = cascade_size

# Plot results: Number of cascades
plt.figure(figsize=(10, 8))
plt.imshow(results_cascade_count, extent=[0, 0.3, 1, 0], aspect='auto', cmap='viridis')
plt.colorbar(label='Number of Cascades')
plt.title('Number of Cascades vs Tau and Theta')
plt.xlabel('Theta (Trading Fee)')
plt.ylabel('Tau (Search Cost)')
plt.show()

# Plot results: Cascade size
plt.figure(figsize=(10, 8))
plt.imshow(results_cascade_size, extent=[0, 0.3, 1, 0], aspect='auto', cmap='viridis')
plt.colorbar(label='Cascade Size')
plt.title('Cascade Size vs Tau and Theta')
plt.xlabel('Theta (Trading Fee)')
plt.ylabel('Tau (Search Cost)')
plt.show()