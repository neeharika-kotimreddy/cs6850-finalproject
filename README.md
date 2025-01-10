# Cascade Models with Transaction Costs and Temporal Components

This project extends the classical Watts threshold model by incorporating transaction costs and temporal components to analyze cascading behavior in networks. The study provides insights into the impact of these factors on cascade frequency, size, and nature, with applications in domains such as financial markets and social networks.


## Description

Information cascades occur when individuals imitate others' behaviors, often ignoring their private signals. This project extends the Watts threshold model by introducing two real-world factors:
- **Transaction Costs**: These include trading fees and search costs that influence economic decisions.
- **Temporal Components**: These represent dynamic interactions over time, where connections between individuals activate and deactivate intermittently.

The model simulates cascades across three network topologies:
1. Random Networks
2. Scale-Free Networks
3. Spatially-Clustered Networks

Key metrics, such as gross information cascades and trend shift cascades, are used to evaluate the impact of these factors on cascading behavior. The findings reveal the interplay between economic constraints and temporal dynamics, providing a nuanced understanding of cascade propagation in networks.

---

## Getting Started

### Dependencies

To run the simulations, the following prerequisites are required:
- **Python 3.x** 
- Libraries:
  - `numpy`
  - `matplotlib`
  - `networkx`
  - `random`
  - `scipy`


### Installing

1. Clone the repository.
2. Navigate to the project directory.
3. Install the required libraries.

### Executing Program
1. Run the simulation script:
   ```bash
    python simulate_cascades.py
2. Modify simulation parameters such as pedge, theta, and tau in the script to explore different scenarios.

---

## Authors
* Neeharika Kotimreddy
* Sanjana Nandi
