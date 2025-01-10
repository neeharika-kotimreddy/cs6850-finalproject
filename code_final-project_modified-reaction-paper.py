import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import tqdm
from tqdm import tqdm

# Create network (choose topology: "random", "scale_free", "clustered")
def create_network(topology="random"):
    if topology == "random":
        return nx.erdos_renyi_graph(N, 0.1)
    elif topology == "scale_free":
        return nx.barabasi_albert_graph(N, 3)
    elif topology == "clustered":
        return nx.powerlaw_cluster_graph(N, 3, 0.1)

# Initialize trading fees on nodes and search fees on edges
def initialize_transaction_costs(G):
    for u, v in G.edges():
        G[u][v]['search_fee'] = max_search_fee # random.uniform(0, max_search_fee)
    for node in G.nodes():
        G.nodes[node]['trading_fee'] = max_trading_fee # random.uniform(0, max_trading_fee) 

# Initialize trade actions with a mix of "buy", "sell", and "hold"
def initialize_trade_actions(G):
    nx.set_node_attributes(G, "hold", "trade_action")  # Default all nodes to "hold"

    initial_buy_ratio = 0.1         # Initial proportion of nodes set to "buy"
    initial_sell_ratio = 0.1        # Initial proportion of nodes set to "sell"

    # Select random nodes for initial "buy" and "sell" actions
    buy_nodes = random.sample(G.nodes(), int(initial_buy_ratio * N))
    sell_nodes = random.sample(set(G.nodes()) - set(buy_nodes), int(initial_sell_ratio * N))

    for node in buy_nodes:
        G.nodes[node]["trade_action"] = "buy"
    for node in sell_nodes:
        G.nodes[node]["trade_action"] = "sell"

# Decision-making function with trading fees, search fees, and asset price
def decide_trade_action(G, node, price):
    neighbors = list(G.neighbors(node))
    trading_fee = G.nodes[node].get('trading_fee', 0) 
    search_fee_sum = sum(G[node][neighbor]['search_fee'] for neighbor in neighbors if G[node][neighbor].get('is_active', False))

    # Estimate potential profit by anticipating a multiplier based on neighbor actions
    buy_price = price + trading_fee + search_fee_sum # Effective price after fees
    anticipated_sell_price = price * 1.1  # 10% increase based on buy influence
    buy_potential = (anticipated_sell_price - buy_price)  # Future profit if buying

    sell_price = price - trading_fee - search_fee_sum # Effective profit after fees
    anticipated_buy_price = price * 0.9  # 10% decrease based on sell influence
    sell_potential = (sell_price - anticipated_buy_price)  # Future profit if selling
    
    # Decision logic based on highest potential action
    if buy_potential > sell_potential and buy_potential > minimum_potential:
        return "buy"
    elif sell_potential > buy_potential and sell_potential > minimum_potential:
        return "sell"
    else:
        # return "hold"
        return G[node]["trade_action"]

# Run the simulation with temporal component and track metrics
def run_simulation(G, price):
    global gross_buy_cascades, gross_sell_cascades, gross_hold_cascades, trend_shift_buy, trend_shift_sell, trend_shift_hold
    previous_buy_count = 0
    previous_sell_count = 0
    previous_hold_count = 0

    for timestep in range(timesteps):
        # print(f"Timestep {timestep+1}")
        
        # Temporarily activate edges based on probability
        for u, v in G.edges():
            G[u][v]['is_active'] = random.random() < edge_active_prob

        # Determine trade actions for each node
        buy_count = 0
        sell_count = 0
        hold_count = 0

        for node in G.nodes():
            trade_action = decide_trade_action(G, node, price)
            G.nodes[node]["trade_action"] = trade_action
            
            if trade_action == "buy":
                gross_buy_cascades += 1
                buy_count += 1
            elif trade_action == "sell":
                gross_sell_cascades += 1
                sell_count += 1
            elif trade_action == "hold":
                gross_hold_cascades += 1
                hold_count += 1

        # Measure Trend Shift Cascades (market volatility) by tracking changes in buy/sell actions
        if abs(buy_count - previous_buy_count) > N * 0.05:
            trend_shift_buy += 1
        if abs(sell_count - previous_sell_count) > N * 0.05:
            trend_shift_sell += 1
        if abs(hold_count - previous_hold_count) > N * 0.05:
            trend_shift_hold += 1

        # Update previous counts
        previous_buy_count = buy_count
        previous_sell_count = sell_count
        previous_hold_count = hold_count

    # Print summary of metrics
    if to_print == "yes":
        print("\n--- Simulation Summary ---")
        print(f"Probability that Edge is Active: {prob}")
        print(f"Asset Price: {price}")
        print(f"Max Possible Trading Fee: {trading_fee}")
        print(f"Max Possible Search Fee: {search_fee}")
        print(f"Gross Buy Cascades: {gross_buy_cascades}")
        print(f"Gross Sell Cascades: {gross_sell_cascades}")
        print(f"Gross Hold Cascades: {gross_hold_cascades}")
        print(f"Trend Shift Buy Cascades: {trend_shift_buy}")
        print(f"Trend Shift Sell Cascades: {trend_shift_sell}")
        print(f"Trend Shift Hold Cascades: {trend_shift_hold}")
    elif to_print == "no":
        with open(f"final-project/{exp_name}/output.txt", "a") as file:
            lines = []
            lines.append("--- Simulation Summary ---")
            lines.append(f"Probability that Edge is Active: {prob}")
            lines.append(f"Asset Price: {price}")
            lines.append(f"Max Possible Trading Fee: {trading_fee}")
            lines.append(f"Max Possible Search Fee: {search_fee}")
            lines.append(f"Gross Buy Cascades: {gross_buy_cascades}")
            lines.append(f"Gross Sell Cascades: {gross_sell_cascades}")
            lines.append(f"Gross Hold Cascades: {gross_hold_cascades}")
            lines.append(f"Trend Shift Buy Cascades: {trend_shift_buy}")
            lines.append(f"Trend Shift Sell Cascades: {trend_shift_sell}")
            lines.append(f"Trend Shift Hold Cascades: {trend_shift_hold}")
            lines.append(f"\n")
            file.writelines(f"{path}\n" for path in lines) 
    return G

asset_price = 30
probs = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
max_trading_fees = list(range(asset_price))
max_search_fees = list(range(asset_price))
max_trading_fees = [t/30 for t in max_trading_fees]
max_search_fees = [s/30 for s in max_search_fees]
gross_cascades = {}
trend_shift_cascades = {}

min_t_fee = asset_price
min_s_fee = asset_price
max_t_fee = 0
max_s_fee = 0

exp_name = input("What is the run name?\n")
if not os.path.exists(os.path.join(os.getcwd(), f"final-project/{exp_name}")):
    os.mkdir(f"final-project/{exp_name}")
print("")

to_print = input("Should I print? (yes/no)\n")
while to_print != "yes" and to_print != "no":
    to_print = input("Please input yes or no.\n")

with open("code_final-project.py", "r") as python_file:
    with open(f"final-project/{exp_name}/output.txt", "a") as text_file:
        text_file.write("Python file at creation: ")
        text_file.write(python_file.read())
        text_file.write("\n")

with tqdm(total=len(probs) * len(max_trading_fees) * len(max_search_fees)) as pbar:
    for prob in probs: 
        gross_cascades[prob] = np.zeros((len(max_trading_fees), len(max_search_fees)))
        trend_shift_cascades[prob] = np.zeros((len(max_trading_fees), len(max_search_fees)))
        for t in range(len(max_trading_fees)):
            trading_fee = max_trading_fees[t]
            for s in range(len(max_search_fees)):
                search_fee = max_search_fees[s]
                # if asset_price - trading_fee >= search_fee:
                if True:                
                    # Parameters
                    N = 100                         # Number of nodes
                    minimum_potential = 0.5         # Threshold for cascade activation
                    timesteps = 20                  # Number of timesteps in the simulation
                    edge_active_prob = prob         # Probability that an edge exists at a given timestep
                    max_trading_fee = trading_fee   # Maximum trading fee per trade
                    max_search_fee = search_fee     # Maximum search fee for deciding on a trade
                    
                    max_t_fee = max(max_t_fee, max_trading_fee)
                    max_s_fee = max(max_s_fee, max_search_fee)
                    min_t_fee = min(min_t_fee, max_trading_fee)
                    min_s_fee = min(min_s_fee, max_search_fee)

                    # Metrics
                    gross_buy_cascades = 0
                    gross_sell_cascades = 0
                    gross_hold_cascades = 0
                    trend_shift_buy = 0
                    trend_shift_sell = 0
                    trend_shift_hold = 0
                    previous_buy_count = 0
                    previous_sell_count = 0
                    previous_hold_count = 0

                    # Main
                    topology = "random"  # Choose "random", "scale_free", or "clustered"
                    G = create_network(topology)
                    initialize_transaction_costs(G)
                    initialize_trade_actions(G)
                    final_graph = run_simulation(G, asset_price)

                    gross_cascades[prob][t, s] = gross_buy_cascades + gross_sell_cascades + gross_hold_cascades
                    trend_shift_cascades[prob][t, s] = trend_shift_buy + trend_shift_sell + trend_shift_hold
                else:
                    gross_cascades[prob][t, s] = -1 
                    trend_shift_cascades[prob][t, s] = -1

                pbar.update(1)

temp_g = np.concatenate(list(gross_cascades.values()))
temp_t = np.concatenate(list(trend_shift_cascades.values()))
g_min = np.min(temp_g)
g_max = np.max(temp_g)
t_min = np.min(temp_t)
t_max = np.max(temp_t)

for prob in probs:
    # # Plot each metric as a heatmap
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))

    # Gross Cascades
    im1 = axs[0].imshow(gross_cascades[prob], aspect='auto', origin='lower', vmin=g_min, vmax=g_max, cmap='viridis',
                        extent=[min_t_fee, max_t_fee, min_s_fee, max_s_fee])
    axs[0].set_title(f'Gross Cascades with Edge Probability {prob}')
    axs[0].set_xlabel('Max Trading Fee')
    axs[0].set_ylabel('Max Search Fee')
    fig.colorbar(im1, ax=axs[0], orientation='vertical')

    # Trend Shift Buy Cascades
    im2 = axs[1].imshow(trend_shift_cascades[prob], aspect='auto', origin='lower', vmin=t_min, vmax=t_max, cmap='viridis',
                        extent=[min_t_fee, max_t_fee, min_s_fee, max_s_fee])
    axs[1].set_title(f"Trend Shift Cascades with Edge Probability {prob}")
    axs[1].set_xlabel('Max Trading Fee')
    axs[1].set_ylabel('Max Search Fee')
    fig.colorbar(im2, ax=axs[1], orientation='vertical')
    
    plt.tight_layout()
    
    plt.savefig(f"final-project/{exp_name}/prob{prob:.1f}.png")