import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Create network (choose topology: "random", "scale_free", "clustered")
def create_network(topology="random"):
    if topology == "random":
        return nx.erdos_renyi_graph(N, 0.1)
    elif topology == "scale_free":
        return nx.barabasi_albert_graph(N, 3)
    elif topology == "clustered":
        return nx.powerlaw_cluster_graph(N, 3, 0.1)

# Initialize trading fees and search costs on edges
def initialize_transaction_costs(G):
    for u, v in G.edges():
        G[u][v]['trading_fee'] = random.uniform(0, max_trading_fee)
        G[u][v]['search_cost'] = random.uniform(0, max_search_fee)

# Initialize trade actions with a mix of "buy", "sell", and "hold"
def initialize_trade_actions(G):
    nx.set_node_attributes(G, "hold", "trade_action")  # Default all nodes to "hold"

    # Select random nodes for initial "buy" and "sell" actions
    buy_nodes = random.sample(G.nodes(), int(initial_buy_ratio * N))
    sell_nodes = random.sample(set(G.nodes()) - set(buy_nodes), int(initial_sell_ratio * N))

    for node in buy_nodes:
        G.nodes[node]["trade_action"] = "buy"
    for node in sell_nodes:
        G.nodes[node]["trade_action"] = "sell"

# Decision-making function with trading fees, search costs, and asset price
def decide_trade_action(G, node):
    neighbors = list(G.neighbors(node))
    trading_fee_sum = sum(G[node][neighbor]['trading_fee'] for neighbor in neighbors if G[node][neighbor].get('is_active', False))
    search_cost_sum = sum(G[node][neighbor]['search_cost'] for neighbor in neighbors if G[node][neighbor].get('is_active', False))
    
    # Estimate potential profit by anticipating a multiplier based on neighbor actions
    buy_price = fixed_asset_price - trading_fee_sum  # Effective price after fees
    anticipated_sell_price = fixed_asset_price * 1.1  # 10% increase based on buy influence
    buy_potential = (anticipated_sell_price - buy_price) - search_cost_sum  # Profit if buying

    sell_price = fixed_asset_price + trading_fee_sum  # Effective price after fees
    anticipated_buy_price = fixed_asset_price * 1.1  # 10% decrease based on sell influence
    sell_potential = (sell_price - anticipated_buy_price) - search_cost_sum  # Profit if selling
    
    # Decision logic based on highest potential action
    if buy_potential > sell_potential and buy_potential > threshold:
        return "buy"
    elif sell_potential > buy_potential and sell_potential > threshold:
        return "sell"
    else:
        return "hold"

# Run the simulation with temporal component and track metrics
def run_simulation(G):
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
            trade_action = decide_trade_action(G, node)
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
    print("\n--- Simulation Summary ---")
    print(f"Gross Buy Cascades: {gross_buy_cascades}")
    print(f"Gross Sell Cascades: {gross_sell_cascades}")
    print(f"Gross Hold Cascades: {gross_hold_cascades}")
    print(f"Trend Shift Buy Cascades: {trend_shift_buy}")
    print(f"Trend Shift Sell Cascades: {trend_shift_sell}")
    print(f"Trend Shift Hold Cascades: {trend_shift_hold}")
    return G

fixed_asset_price = 30
probs = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
max_trading_fees = list(range(fixed_asset_price))
max_search_fees = list(range(fixed_asset_price))
gross_cascades = {}
trend_shift_cascades = {}

for prob in probs: 
    gross_cascades[prob] = np.zeros((fixed_asset_price, fixed_asset_price))
    trend_shift_cascades[prob] = np.zeros((fixed_asset_price, fixed_asset_price))
    for trading_fee in max_trading_fees:
        for search_fee in max_search_fees:
            if fixed_asset_price - trading_fee >= search_fee:
                print(f"Trading Fee {trading_fee} Search Cost {search_fee} Prob {prob}")
                
                # Parameters
                N = 100                         # Number of nodes
                initial_buy_ratio = 0.1         # Initial proportion of nodes set to "buy"
                initial_sell_ratio = 0.1        # Initial proportion of nodes set to "sell"
                threshold = 0.5                 # Threshold for cascade activation
                max_trading_fee = trading_fee   # Maximum trading fee per trade
                max_search_fee = search_fee     # Maximum search cost for deciding on a trade
                timesteps = 20                  # Number of timesteps in the simulation
                edge_active_prob = prob         # Probability that an edge is active during a timestep

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
                final_graph = run_simulation(G)

                gross_cascades[prob][trading_fee, search_fee] = gross_buy_cascades + gross_sell_cascades + gross_hold_cascades
                trend_shift_cascades[prob][trading_fee, search_fee] = trend_shift_buy + trend_shift_sell + trend_shift_hold
            else:
                gross_cascades[prob][trading_fee, search_fee] = -1 
                trend_shift_cascades[prob][trading_fee, search_fee] = -1

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
                        extent=[0, fixed_asset_price, 0, fixed_asset_price])
    axs[0].set_title(f'Gross Cascades with Edge Probability {prob}')
    axs[0].set_xlabel('Max Trading Fee')
    axs[0].set_ylabel('Max Search Cost')
    fig.colorbar(im1, ax=axs[0], orientation='vertical')

    # Trend Shift Buy Cascades
    im2 = axs[1].imshow(trend_shift_cascades[prob], aspect='auto', origin='lower', vmin=t_min, vmax=t_max, cmap='viridis',
                        extent=[0, fixed_asset_price, 0, fixed_asset_price])
    axs[1].set_title(f"Trend Shift Cascades with Edge Probability {prob}")
    axs[1].set_xlabel('Max Trading Fee')
    axs[1].set_ylabel('Max Search Cost')
    fig.colorbar(im2, ax=axs[1], orientation='vertical')
    
    plt.tight_layout()
    
    if not os.path.exists(os.path.join(os.getcwd(), "reaction-paper/reaction-paper-model-4")):
        os.mkdir("reaction-paper/reaction-paper-model-4")
    plt.savefig(f"reaction-paper/reaction-paper-model-4/{prob:.1f}.png")