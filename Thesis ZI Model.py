# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 14:11:50 2025

@author: mjmde

"""


# =============================================================================
# -------------------------------------------------------------------------
# Zero-Intelligence SIMULATION
# -------------------------------------------------------------------------
# =============================================================================
##############################################################################
# Zero Intelligence Simulation for NON-fractional Inventory
##############################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import floor
import os
import concurrent.futures
import warnings
from scipy.stats import linregress

# =============================================================================
# Suppress divide-by-zero warnings from NumPy
# =============================================================================
warnings.filterwarnings("ignore", category=RuntimeWarning)
np.seterr(divide='ignore', invalid='ignore')

# =============================================================================
# Helper Function: Run a Single Simulation (Modified with first_layer)
# =============================================================================
def run_simulation(sim_seed, ini_inventory, time_end=600, switch_random=1, num_echelons=10,
                   inventory_delay=1, request_delay=0.01, max_request_interval=4,
                   endowment=30, price_sell=10, price_request=0, price_inventory=2,
                   STOCKOUT_COST=0, first_layer="symmetrical"):
    """
    Runs one simulation with the specified parameters.
    The 'first_layer' parameter controls the initial inventory distribution:
      - "symmetrical": Each agent receives ini_inventory.
      - "random": The total inventory is randomly partitioned among agents.
      - "first_only": The first agent gets all the inventory.
    Returns a DataFrame with the simulation event log.
    """
    np.random.seed(sim_seed)  # Set unique random seed for reproducibility

    if first_layer == "symmetrical":
        inventory = np.full(num_echelons, ini_inventory, dtype=float)
    elif first_layer == "random":
        total_inventory = ini_inventory * num_echelons
        inventory = np.random.multinomial(total_inventory, [1/num_echelons]*num_echelons).astype(float)
    elif first_layer == "first_only":
        inventory = np.zeros(num_echelons, dtype=float)
        inventory[0] = ini_inventory * num_echelons
    else:
        raise ValueError("Invalid first_layer parameter. Choose 'symmetrical', 'random', or 'first_only'.")

    balance = np.full(num_echelons, endowment, dtype=float)
    requests_count = np.zeros(num_echelons, dtype=int)
    
    if switch_random == 1:
        next_request_time = np.full(num_echelons, request_delay) + np.random.uniform(0, max_request_interval, num_echelons)
    else:
        next_request_time = np.full(num_echelons, request_delay) + (max_request_interval / 2)
    
    next_inventory_time = inventory_delay
    current_time = 0.0
    
    event_log = []
    
    # Log initial state for each agent.
    for i in range(num_echelons):
        event_log.append({
            "Time Step": 0,
            "Agent": i + 1,
            "Requested": 0,
            "Collected": 0,
            "Completed": 0,
            "Inventory": inventory[i],
            "Earnings": 0,
            "Costs requested": 0,
            "Costs Inventory": 0,
            "Costs stockout": 0,
            "Net Change": 0,
            "Balance": balance[i],
            "Bankrupt": False
        })
    
    while current_time < time_end:
        next_event_time = min(next_inventory_time, next_request_time.min())
        if next_event_time > time_end:
            current_time = time_end
            break
        current_time = next_event_time
        
        if abs(current_time - next_inventory_time) < 1e-8:
            for i in range(num_echelons):
                cost = inventory[i] * price_inventory
                balance[i] -= cost
                event_log.append({
                    "Time Step": current_time,
                    "Agent": i + 1,
                    "Requested": 0,
                    "Collected": 0,
                    "Completed": 0,
                    "Inventory": inventory[i],
                    "Earnings": 0,
                    "Costs requested": 0,
                    "Costs Inventory": -cost,
                    "Costs stockout": 0,
                    "Net Change": -cost,
                    "Balance": balance[i],
                    "Bankrupt": (balance[i] < -(inventory[i] * price_sell) + price_request)
                })
            next_inventory_time += inventory_delay
            
        else:
            candidate = int(np.argmin(next_request_time))
            requests_count[candidate] += 1
            balance[candidate] -= price_request
            
            pred = candidate - 1 if candidate != 0 else num_echelons - 1
            
            if inventory[pred] > 0:
                inventory[candidate] += 1
                inventory[pred] -= 1
                balance[pred] += price_sell
                event_log.append({
                    "Time Step": current_time,
                    "Agent": candidate + 1,
                    "Requested": 1,
                    "Collected": 1,
                    "Completed": 0,
                    "Inventory": inventory[candidate],
                    "Earnings": 0,
                    "Costs requested": -price_request,
                    "Costs Inventory": 0,
                    "Costs stockout": 0,
                    "Net Change": -price_request,
                    "Balance": balance[candidate],
                    "Bankrupt": (balance[candidate] < -(inventory[candidate] * price_sell) + price_request)
                })
                event_log.append({
                    "Time Step": current_time,
                    "Agent": pred + 1,
                    "Requested": 0,
                    "Collected": 0,
                    "Completed": 1,
                    "Inventory": inventory[pred],
                    "Earnings": price_sell,
                    "Costs requested": 0,
                    "Costs Inventory": 0,
                    "Costs stockout": 0,
                    "Net Change": price_sell,
                    "Balance": balance[pred],
                    "Bankrupt": (balance[pred] < -(inventory[pred] * price_sell) + price_request)
                })
            else:
                event_log.append({
                    "Time Step": current_time,
                    "Agent": candidate + 1,
                    "Requested": 1,
                    "Collected": 0,
                    "Completed": 0,
                    "Inventory": inventory[candidate],
                    "Earnings": 0,
                    "Costs requested": -price_request,
                    "Costs Inventory": 0,
                    "Costs stockout": -STOCKOUT_COST,
                    "Net Change": -(price_request + STOCKOUT_COST),
                    "Balance": balance[candidate],
                    "Bankrupt": (balance[candidate] < -(inventory[candidate] * price_sell) + price_request)
                })
            
            if switch_random == 1:
                next_request_time[candidate] = current_time + request_delay + np.random.uniform(0, max_request_interval)
            else:
                next_request_time[candidate] = current_time + request_delay + (max_request_interval / 2)
            
    event_log_df = pd.DataFrame(event_log)
    return event_log_df

# =============================================================================
# Helper Function: Plot Event Log Per Agent (Multi-Subplot) WITHOUT Showing
# =============================================================================
def plot_event_log_multisubplot(event_log_df, save_path=None):
    participants = sorted(event_log_df["Agent"].unique())
    num_participants = len(participants)
    fig, axs = plt.subplots(num_participants, 1, figsize=(11.7, 8.3), sharex=True)
    if num_participants == 1:
        axs = [axs]
    
    max_inv = event_log_df["Inventory"].max()
    max_bal = event_log_df["Balance"].max()
    scale_factor = max_inv / max_bal if max_bal != 0 else 1
    
    for ax, agent_id in zip(axs, participants):
        part_data = event_log_df[event_log_df["Agent"] == agent_id].sort_values("Time Step")
        ax.plot(part_data["Time Step"], part_data["Inventory"], color="blue", label="Inventory")
        ax.plot(part_data["Time Step"], part_data["Balance"] * scale_factor,
                color="orange", label="Balance")
        bankrupt_mask = part_data["Bankrupt"].values.astype(bool)
        times = part_data["Time Step"].values
        bal_scaled = (part_data["Balance"] * scale_factor).values
        
        segments = []
        current_seg_times = []
        current_seg_balance = []
        for i in range(len(bankrupt_mask)):
            if bankrupt_mask[i]:
                current_seg_times.append(times[i])
                current_seg_balance.append(bal_scaled[i])
            else:
                if current_seg_times:
                    segments.append((current_seg_times, current_seg_balance))
                    current_seg_times = []
                    current_seg_balance = []
        if current_seg_times:
            segments.append((current_seg_times, current_seg_balance))
        
        bankrupt_label_used = False
        for seg_times, seg_balance in segments:
            if len(seg_times) == 1:
                seg_times = [seg_times[0], seg_times[0] + 1e-6]
                seg_balance = [seg_balance[0], seg_balance[0]]
            if not bankrupt_label_used:
                ax.plot(seg_times, seg_balance, color="red", linewidth=2, label="Bankrupt")
                bankrupt_label_used = True
            else:
                ax.plot(seg_times, seg_balance, color="red", linewidth=2)
        
        ax.set_title(f"Agent {agent_id}")
        ax.set_ylabel("Inventory")
        secax = ax.secondary_yaxis('right', functions=(lambda x: x / scale_factor, lambda x: x * scale_factor))
        secax.set_ylabel("Balance")
    
    handles, labels = axs[0].get_legend_handles_labels()
    unique = {}
    for handle, label in zip(handles, labels):
        if label not in unique:
            unique[label] = handle
    axs[0].legend(unique.values(), unique.keys(), loc="upper left")
    plt.xlabel("Time (s)")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, format="pdf")
    plt.close(fig)


# =============================================================================
# CLEAN MAIN LOOP — batch runner for supply chain sim
# =============================================================================
if __name__ == '__main__':
    output_dir = "ZI_simulation_outputs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Parameter grid
    first_layer_distributions = ["symmetrical", "random", "first_only"]
    inventory_costs = [0.05, 0.5, 1.0, 1.5, 2.0]
    agent_counts = [4, 5, 6, 10, 15, 20, 40]
    initial_inventories = [1, 2, 3, 4, 5, 10, 20, 100, 1000]

    # Loop over all combinations
    for first_layer in first_layer_distributions:
        first_layer_dir = os.path.join(output_dir, f"FirstLayer_{first_layer}")
        os.makedirs(first_layer_dir, exist_ok=True)

        for inv_cost in inventory_costs:
            cost_dir = os.path.join(first_layer_dir, f"InventoryCost_{inv_cost}")
            os.makedirs(cost_dir, exist_ok=True)

            for num_agents in agent_counts:
                agents_dir = os.path.join(cost_dir, f"Agents_{num_agents}")
                os.makedirs(agents_dir, exist_ok=True)

                for ini_inv in initial_inventories:
                    print(f"\n=== Running simulation: FirstLayer={first_layer}, InventoryCost={inv_cost}, Agents={num_agents}, IniInventory={ini_inv} ===")

                    # Run one simulation
                    sim_log_df = run_simulation(
                        sim_seed=55,
                        ini_inventory=ini_inv,
                        time_end=1000000,  # FULL run 1 million seconds
                        num_echelons=num_agents,
                        price_inventory=inv_cost,
                        first_layer=first_layer
                    )

                    # Save FULL sim log to CSV
                    sim_log_filename = os.path.join(agents_dir, f"SimLog_Agents_{num_agents}_IniInventory_{ini_inv}_InvCost_{inv_cost}.csv")
                    sim_log_df.to_csv(sim_log_filename, index=False)
                    print(f"Saved FULL simulation log to {sim_log_filename}")

                    # Only make plot when num_agents == 5 and of first 600s
                    if num_agents == 5:
                        sim_log_600s = sim_log_df[sim_log_df["Time Step"] <= 600].copy()
                        plot_filename = os.path.join(agents_dir, f"EventLogPlot600s_Agents_{num_agents}_IniInventory_{ini_inv}_InvCost_{inv_cost}.pdf")
                        plot_event_log_multisubplot(sim_log_600s, save_path=plot_filename)
                        print(f"Saved 600s plot to {plot_filename}")        

    print("\nAll batch simulations completed.")
    




# =============================================================================
########################################################################
# LOOP TIME increases with agents and initial inventory
########################################################################
# =============================================================================

# =============================================================================
# Compute LOOP TIMES (Agent 1) for Supply Chain Sim Logs 
# =============================================================================

import os
import pandas as pd

# === CONFIGURATION ===
base_folder          = "ZI_simulation_outputs/FirstLayer_symmetrical/InventoryCost_0.5"
agent_counts         = [4, 5, 6, 10, 15, 20, 40]
ini_inventories      = [1, 2, 3, 4, 5, 10, 20, 100, 1000]
target_agent         = 1
loop_output_folder   = "LoopTimes"

# === MAIN LOOP (chunked reading) ===
for num_agents in agent_counts:
    agents_dir = os.path.join(base_folder, f"Agents_{num_agents}")

    for ini_inv in ini_inventories:
        print(f"\n=== Processing Agents={num_agents}, IniInventory={ini_inv} ===")

        simlog_path = os.path.join(
            agents_dir,
            f"SimLog_Agents_{num_agents}_IniInventory_{ini_inv}_InvCost_0.5.csv"
        )
        if not os.path.exists(simlog_path):
            print(f"[!] Missing file {simlog_path}")
            continue

        # We only need these three columns:
        usecols = ["Time Step", "Agent", "Completed"]

        # We'll accumulate only the timestamps when Agent==1 AND Completed==1
        # Because we only care about those events to compute loop times.
        agent1_times = []  # List[float]

        # Read the CSV in chunks of, say, 1 million rows at a time
        chunk_iter = pd.read_csv(
            simlog_path,
            usecols=usecols,
            chunksize=1_000_000  # adjust based on RAM; 1e6 is usually safe
        )
        for chunk in chunk_iter:
            # Filter immediately in each chunk
            mask = (chunk["Agent"] == target_agent) & (chunk["Completed"] == 1)
            if mask.any():
                # Extend our list with exactly those “Time Step” values
                agent1_times.extend(chunk.loc[mask, "Time Step"].tolist())

        total_completed = len(agent1_times)

        # === Corrected: use TOTAL inventory, not ini_inv ===
        TOTAL_inventory = ini_inv * num_agents
        required = TOTAL_inventory * 2

        if total_completed < required:
            print(f"[!] Too few completed‐sale events for Agent 1 "
                  f"(have {total_completed}, need at least {required}) → skipping")
            continue

        # Now compute the loop times.  Because the streaming‐to‐CSV already wrote 
        # rows in ascending “Time Step” order, agent1_times is already sorted.
        loop_times = []

        # === Corrected: batch size = TOTAL_inventory ===
        n_full_blocks = total_completed // TOTAL_inventory  # integer division

        for loop_idx in range(1, n_full_blocks):
            start_idx = (loop_idx - 1) * TOTAL_inventory
            end_idx   = loop_idx * TOTAL_inventory
            t_start = agent1_times[start_idx]
            t_end   = agent1_times[end_idx]
            loop_times.append({
                "Loop_Index": loop_idx,
                "Loop_Time": t_end - t_start
            })

        # Build a tiny DataFrame—and write it out
        loop_df = pd.DataFrame(loop_times, columns=["Loop_Index", "Loop_Time"])

        loop_output_dir = os.path.join(agents_dir, loop_output_folder)
        os.makedirs(loop_output_dir, exist_ok=True)

        out_csv_path = os.path.join(
            loop_output_dir,
            f"LoopTimes_Agents_{num_agents}_IniInventory_{ini_inv}.csv"
        )
        loop_df.to_csv(out_csv_path, index=False)
        print(f" → Saved LoopTimes CSV: {out_csv_path}")

print("\nAll loop‐time files generated!")





# =============================================================================
# Plot LoopTimes Summary: Mean and StdDev per Agents and Inventory 
# =============================================================================

import os
import pandas as pd
import matplotlib.pyplot as plt

# === Matplotlib configuration for scientific style ===
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "axes.edgecolor": "black",
    "axes.linewidth": 1.0,
    "lines.linewidth": 1.5,
    "text.usetex": False
})

# === CONFIGURATION ===
base_folder          = "ZI_simulation_outputs/FirstLayer_symmetrical/InventoryCost_0.5"
agent_counts         = [4, 5, 6, 10, 15, 20, 40]
ini_inventories      = [1, 2, 3, 4, 5, 10, 20, 100, 1000]
loop_output_folder   = "LoopTimes"

# === Prepare storage for plotting ===
# Dict of {inventory X: list of mean loop times per N}
mean_loop_times = {X: [] for X in ini_inventories}
std_loop_times  = {X: [] for X in ini_inventories}

# === Loop over all N and X combinations ===
for X in ini_inventories:
    for N in agent_counts:
        loop_csv_path = os.path.join(
            base_folder,
            f"Agents_{N}",
            loop_output_folder,
            f"LoopTimes_Agents_{N}_IniInventory_{X}.csv"
        )
        
        if not os.path.exists(loop_csv_path):
            print(f"[!] Missing LoopTimes CSV → skipping: N={N}, X={X}")
            mean_loop_times[X].append(float("nan"))
            std_loop_times[X].append(float("nan"))
            continue
        
        # Load LoopTimes CSV
        df_loop = pd.read_csv(loop_csv_path)
        
        # Compute mean and stddev of Loop_Time
        mean_val = df_loop["Loop_Time"].mean()
        std_val  = df_loop["Loop_Time"].std()

        # Store in dict
        mean_loop_times[X].append(mean_val)
        std_loop_times[X].append(std_val)

# === Plotting ===
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

# Panel (a) — Mean loop time
ax1 = axes[0]
for X in ini_inventories:
    ax1.plot(agent_counts, mean_loop_times[X], label=f"IniInv={X}")
ax1.set_xlabel("Number of Agents")
ax1.set_ylabel("Mean Loop Time")
ax1.set_title("a) Mean Loop Time vs. Agents")
ax1.grid(True, linestyle='--', linewidth=0.5)
ax1.legend(title="Initial Inventory", fontsize=9)

# Panel (b) — Stddev loop time
ax2 = axes[1]
for X in ini_inventories:
    ax2.plot(agent_counts, std_loop_times[X], label=f"IniInv={X}")
ax2.set_xlabel("Number of Agents")
ax2.set_ylabel("StdDev of Loop Time")
ax2.set_title("b) StdDev of Loop Time vs. Agents")
ax2.grid(True, linestyle='--', linewidth=0.5)
ax2.legend(title="Initial Inventory", fontsize=9)

# === Final layout and save ===
plt.tight_layout()

output_plot_path = os.path.join(base_folder, "LoopTimes_Summary_Mean_StdDev.png")
plt.savefig(output_plot_path, dpi=300)  
plt.close()

print(f"\n Summary plot saved to: {output_plot_path}")



# =============================================================================
# Plot LoopTimes Summary: Mean and StdDev — Inventory as X, Agents as Legend
# =============================================================================

import os
import pandas as pd
import matplotlib.pyplot as plt

# === Matplotlib configuration for scientific style ===
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "axes.edgecolor": "black",
    "axes.linewidth": 1.0,
    "lines.linewidth": 1.5,
    "text.usetex": False
})

# === CONFIGURATION ===
base_folder          = "ZI_simulation_outputs/FirstLayer_symmetrical/InventoryCost_0.5"
agent_counts         = [4, 5, 6, 10, 15, 20, 40]
ini_inventories      = [1, 2, 3, 4, 5, 10, 20, 100, 1000]
loop_output_folder   = "LoopTimes"

# === Prepare storage for plotting ===
# Dict of {agents N: list of mean loop times per inventory X}
mean_loop_times = {N: [] for N in agent_counts}
std_loop_times  = {N: [] for N in agent_counts}

# === Loop over all N and X combinations ===
for N in agent_counts:
    for X in ini_inventories:
        loop_csv_path = os.path.join(
            base_folder,
            f"Agents_{N}",
            loop_output_folder,
            f"LoopTimes_Agents_{N}_IniInventory_{X}.csv"
        )
        
        if not os.path.exists(loop_csv_path):
            print(f"[!] Missing LoopTimes CSV → skipping: N={N}, X={X}")
            mean_loop_times[N].append(float("nan"))
            std_loop_times[N].append(float("nan"))
            continue
        
        # Load LoopTimes CSV
        df_loop = pd.read_csv(loop_csv_path)
        
        # Compute mean and stddev of Loop_Time
        mean_val = df_loop["Loop_Time"].mean()
        std_val  = df_loop["Loop_Time"].std()

        # Store in dict
        mean_loop_times[N].append(mean_val)
        std_loop_times[N].append(std_val)

# === Plotting ===
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

# Panel (a) — Mean loop time
ax1 = axes[0]
for N in agent_counts:
    ax1.plot(ini_inventories, mean_loop_times[N], label=f"Agents={N}")
ax1.set_xlabel("Initial Inventory")
ax1.set_ylabel("Mean Loop Time")
ax1.set_title("a) Mean Loop Time vs. Initial Inventory")
ax1.grid(True, linestyle='--', linewidth=0.5)
ax1.legend(title="Number of Agents", fontsize=9)

# Panel (b) — Stddev loop time
ax2 = axes[1]
for N in agent_counts:
    ax2.plot(ini_inventories, std_loop_times[N], label=f"Agents={N}")
ax2.set_xlabel("Initial Inventory")
ax2.set_ylabel("StdDev of Loop Time")
ax2.set_title("b) StdDev of Loop Time vs. Initial Inventory")
ax2.grid(True, linestyle='--', linewidth=0.5)
ax2.legend(title="Number of Agents", fontsize=9)

# === Final layout and save ===
plt.tight_layout()

output_plot_path = os.path.join(base_folder, "LoopTimes_Summary_Inventory_X.png")
plt.savefig(output_plot_path, dpi=300)
plt.close()

print(f"\n Summary plot saved to: {output_plot_path}")




# =============================================================================
# Plot Loop Time Variability: 4 Configurations, Loop Index 50–100
# Scientific Style: Monochrome lines in different greys + small markers
# =============================================================================

import os
import pandas as pd
import matplotlib.pyplot as plt

# === Matplotlib configuration for scientific style ===
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "axes.edgecolor": "black",
    "axes.linewidth": 1.0,
    "lines.linewidth": 1.5,
    "text.usetex": False
})

# === CONFIGURATION ===
base_folder        = "ZI_simulation_outputs/FirstLayer_symmetrical/InventoryCost_0.5"
loop_output_folder = "LoopTimes"
output_plot_path   = os.path.join(base_folder, "LoopTimes_Variability_4Configs_Index50to100_ScientificGreyMarkers.png")

# Target configs: (agents, inventory)
target_configs = [
    (5, 1),
    (5, 20),
    (40, 1),
    (40, 20)
]

# Marker styles for 4 lines — SCIENTIFIC convention
marker_styles = ['^', 's', 'x', 'o']  # triangle, square, cross, circle

# Grey shades for 4 lines
grey_colors = ['#000000', '#333333', '#666666', '#999999']

# === Load loop times for each config ===
loop_data = {}

for (N, X) in target_configs:
    loop_csv_path = os.path.join(
        base_folder,
        f"Agents_{N}",
        loop_output_folder,
        f"LoopTimes_Agents_{N}_IniInventory_{X}.csv"
    )
    
    if not os.path.exists(loop_csv_path):
        print(f"[!] Missing LoopTimes CSV → skipping: Agents={N}, IniInventory={X}")
        continue
    
    # Load LoopTimes CSV
    df_loop = pd.read_csv(loop_csv_path)
    
    # Extract Loop_Time column as list
    all_loop_times = df_loop["Loop_Time"].tolist()
    
    if len(all_loop_times) < 100:
        print(f"[!] Warning: only {len(all_loop_times)} loop times available for Agents={N}, IniInventory={X} → skipping")
        continue
    
    # Take loop index 50 to 100 → Python slice [50:100] → elements 50..99
    selected_loop_times = all_loop_times[50:100]
    
    # Store in dict
    loop_data[(N, X)] = selected_loop_times

# === Plotting ===
fig, ax = plt.subplots(figsize=(7, 5))  # Single plot

# X-axis = 51..100 → for each config
for idx, ((N, X), loop_times) in enumerate(loop_data.items()):
    x_values = list(range(51, 51 + len(loop_times)))  # Loop index 51 to 100
    label = f"Agents={N}, IniInv={X}"
    ax.plot(
        x_values,
        loop_times,
        label=label,
        color=grey_colors[idx % len(grey_colors)],  # different grey shades
        marker=marker_styles[idx % len(marker_styles)],  # different marker shapes
        markersize=4,                    # smaller marker size
        markeredgewidth=0.8,
        markerfacecolor='white',
        markeredgecolor=grey_colors[idx % len(grey_colors)],  # match line color
        linewidth=1.5,
        linestyle='-'  # solid line
    )

# === Axes, grid, legend
ax.set_xlabel("Loop Index (50 to 100)")
ax.set_ylabel("Loop Time")
ax.set_title("Loop Times of Selected Configurations (Index 50–100)")
ax.grid(True, linestyle='--', linewidth=0.5)
ax.legend(fontsize=10)

# === Final layout and save ===
plt.tight_layout()
plt.savefig(output_plot_path, dpi=600)  # High quality for publication
plt.close()

print(f"\n Variability plot saved to: {output_plot_path}")



# =============================================================================
# Plot Mean vs Variance of Loop Times — Scientific Scatter Plot Monochrome + Regression + CI + Filtering
# =============================================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

# === Matplotlib configuration for scientific style ===
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "axes.edgecolor": "black",
    "axes.linewidth": 1.0,
    "lines.linewidth": 1.5,
    "text.usetex": False
})

# === CONFIGURATION ===
base_folder          = "ZI_simulation_outputs/FirstLayer_symmetrical/InventoryCost_0.5"
agent_counts         = [4, 5, 6, 10, 15, 20, 40]
ini_inventories      = [1, 2, 3, 4, 5, 10, 20, 100, 1000]
loop_output_folder   = "LoopTimes"

# === Set cutoff values ===
max_mean = 2000      # adjust this as needed
max_variance = 8000  # adjust this as needed

# === Prepare storage ===
mean_vs_variance_data = []  # List of (mean, variance)

# === Loop over all N and X combinations ===
for X in ini_inventories:
    for N in agent_counts:
        loop_csv_path = os.path.join(
            base_folder,
            f"Agents_{N}",
            loop_output_folder,
            f"LoopTimes_Agents_{N}_IniInventory_{X}.csv"
        )
        
        if not os.path.exists(loop_csv_path):
            print(f"[!] Missing LoopTimes CSV → skipping: N={N}, X={X}")
            continue
        
        # Load LoopTimes CSV
        df_loop = pd.read_csv(loop_csv_path)
        
        # Compute mean and variance of Loop_Time
        mean_val = df_loop["Loop_Time"].mean()
        variance_val = df_loop["Loop_Time"].var()  # Variance
        
        # Store for scatter plot only if under cutoff
        if mean_val < max_mean and variance_val < max_variance:
            mean_vs_variance_data.append((mean_val, variance_val))

# === Convert to numpy arrays
mean_vals = np.array([x[0] for x in mean_vs_variance_data])
variance_vals = np.array([x[1] for x in mean_vs_variance_data])

# === Perform linear regression
slope, intercept, r_value, p_value, std_err = linregress(mean_vals, variance_vals)

# Predicted variance values for regression line
x_fit = np.linspace(mean_vals.min(), mean_vals.max(), 100)
y_fit = intercept + slope * x_fit



# === Plotting ===
fig, ax = plt.subplots(figsize=(7, 5), facecolor='white')
ax.set_facecolor('whitesmoke')

# Plot all points — dark grey dots
ax.scatter(mean_vals, variance_vals, color='#444444', s=40, alpha=0.9)

# Plot regression line — black solid
ax.plot(x_fit, y_fit, color='black', linewidth=1.5, label=f"Regression (R² = {r_value**2:.3f})")



# === Axes, grid, legend
ax.set_xlabel("Mean Loop Time")
ax.set_ylabel("Variance of Loop Time")
ax.set_title("Mean vs Variance of Loop Times (Filtered)")
ax.grid(True, linestyle='--', linewidth=0.5)
ax.legend(fontsize=10)

# === Final layout and save ===
plt.tight_layout()

output_plot_path = os.path.join(base_folder, "LoopTimes_Mean_vs_Variance_Filtered.png")
plt.savefig(output_plot_path, dpi=600)
plt.close()

print(f"\n Mean vs Variance plot saved to: {output_plot_path}")



# =============================================================================
# One 4-panel figure: LoopTimes Summary + Variability + Mean vs Variance
# Panels: a), b), c), d)
# =============================================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

# === Matplotlib configuration for scientific style ===
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "axes.edgecolor": "black",
    "axes.linewidth": 1.0,
    "lines.linewidth": 1.5,
    "text.usetex": False
})

# === CONFIGURATION ===
base_folder        = "ZI_simulation_outputs/FirstLayer_symmetrical/InventoryCost_0.5"
loop_output_folder = "LoopTimes"
output_plot_path   = os.path.join(base_folder, "LoopTimes_FourPanel_Summary.png")

agent_counts         = [4, 5, 6, 10, 15, 20, 40]
ini_inventories      = [1, 2, 3, 4, 5, 10, 20, 100, 1000]

# === Prepare data ===

# For panel a and b
mean_loop_times = {N: [] for N in agent_counts}
std_loop_times  = {N: [] for N in agent_counts}

# For panel c
target_configs = [(5, 1), (5, 20), (40, 1), (40, 20)]
marker_styles = ['^', 's', 'x', 'o']
grey_colors = ['#000000', '#333333', '#666666', '#999999']
loop_data_c = {}

# For panel d
max_mean = 2000
max_variance = 8000
mean_vs_variance_data = []

# === Load all data ===
for N in agent_counts:
    for X in ini_inventories:
        loop_csv_path = os.path.join(
            base_folder,
            f"Agents_{N}",
            loop_output_folder,
            f"LoopTimes_Agents_{N}_IniInventory_{X}.csv"
        )
        
        if not os.path.exists(loop_csv_path):
            print(f"[!] Missing LoopTimes CSV → skipping: N={N}, X={X}")
            mean_loop_times[N].append(float("nan"))
            std_loop_times[N].append(float("nan"))
            continue
        
        df_loop = pd.read_csv(loop_csv_path)
        mean_val = df_loop["Loop_Time"].mean()
        std_val  = df_loop["Loop_Time"].std()
        
        mean_loop_times[N].append(mean_val)
        std_loop_times[N].append(std_val)
        
        # For panel d
        variance_val = df_loop["Loop_Time"].var()
        if mean_val < max_mean and variance_val < max_variance:
            mean_vs_variance_data.append((mean_val, variance_val))

# For panel c — variability plot
for (N, X) in target_configs:
    loop_csv_path = os.path.join(
        base_folder,
        f"Agents_{N}",
        loop_output_folder,
        f"LoopTimes_Agents_{N}_IniInventory_{X}.csv"
    )
    
    if not os.path.exists(loop_csv_path):
        print(f"[!] Missing LoopTimes CSV → skipping: Agents={N}, IniInventory={X}")
        continue
    
    df_loop = pd.read_csv(loop_csv_path)
    all_loop_times = df_loop["Loop_Time"].tolist()
    
    if len(all_loop_times) >= 100:
        selected_loop_times = all_loop_times[50:100]
        loop_data_c[(N, X)] = selected_loop_times

# === Prepare panel d regression ===
mean_vals = np.array([x[0] for x in mean_vs_variance_data])
variance_vals = np.array([x[1] for x in mean_vs_variance_data])

slope, intercept, r_value, p_value, std_err = linregress(mean_vals, variance_vals)

x_fit = np.linspace(mean_vals.min(), mean_vals.max(), 100)
y_fit = intercept + slope * x_fit

# === Create one big 4-panel figure ===
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel a
ax = axes[0, 0]
for N in agent_counts:
    ax.plot(ini_inventories, mean_loop_times[N], label=f"Agents={N}")
ax.set_xlabel("Initial Inventory")
ax.set_ylabel("Mean Loop Time")
ax.set_title("a) Mean Loop Time vs. Initial Inventory")
ax.grid(True, linestyle='--', linewidth=0.5)
ax.legend(title="Number of Agents", fontsize=9)


# Panel b
ax = axes[0, 1]
for N in agent_counts:
    ax.plot(ini_inventories, std_loop_times[N], label=f"Agents={N}")
ax.set_xlabel("Initial Inventory")
ax.set_ylabel("StdDev of Loop Time")
ax.set_title("b) StdDev of Loop Time vs. Initial Inventory")
ax.grid(True, linestyle='--', linewidth=0.5)
ax.legend(title="Number of Agents", fontsize=9)


# Panel c
ax = axes[1, 0]
for idx, ((N, X), loop_times) in enumerate(loop_data_c.items()):
    x_values = list(range(51, 51 + len(loop_times)))
    label = f"Agents={N}, IniInv={X}"
    ax.plot(
        x_values,
        loop_times,
        label=label,
        color=grey_colors[idx % len(grey_colors)],
        marker=marker_styles[idx % len(marker_styles)],
        markersize=4,
        markeredgewidth=0.8,
        markerfacecolor='white',
        markeredgecolor=grey_colors[idx % len(grey_colors)],
        linewidth=1.5,
        linestyle='-'
    )
ax.set_xlabel("Loop Index (50 to 100)")
ax.set_ylabel("Loop Time")
ax.set_title("c) Loop Times of Selected Configurations (Index 50–100)")
ax.grid(True, linestyle='--', linewidth=0.5)
ax.legend(fontsize=9)


# Panel d
ax = axes[1, 1]
ax.scatter(mean_vals, variance_vals, color='#444444', s=40, alpha=0.9)
ax.plot(x_fit, y_fit, color='black', linewidth=1.5, label=f"Regression (R² = {r_value**2:.3f})")
ax.set_xlabel("Mean Loop Time")
ax.set_ylabel("Variance of Loop Time")
ax.set_title("d) Mean vs Variance of Loop Times")
ax.grid(True, linestyle='--', linewidth=0.5)
ax.legend(fontsize=9)


# === Final layout and save ===
plt.tight_layout()
plt.savefig(output_plot_path, dpi=600)
plt.close()

print(f"\n Four-panel figure saved to: {output_plot_path}")





# =============================================================================
############################################################################
# CALIBRATION
############################################################################
# =============================================================================

##############################################################################
# Zero Intelligence Simulation for NON-fractional Inventory 
##############################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import floor
import os
import warnings
from scipy.stats import linregress

warnings.filterwarnings("ignore", category=RuntimeWarning)
np.seterr(divide='ignore', invalid='ignore')

def run_simulation(sim_seed, ini_inventory, time_end=180, switch_random=1, num_echelons=10,
                    inventory_delay=1, request_delay=0.01, max_request_interval=4,
                    endowment=30, price_sell=10, price_request=0, price_inventory=2,
                    STOCKOUT_COST=0, first_layer="symmetrical"):
    np.random.seed(sim_seed)
    
    if first_layer == "symmetrical":
        inventory = np.full(num_echelons, ini_inventory, dtype=float)
    elif first_layer == "random":
        total_inventory = ini_inventory * num_echelons
        inventory = np.random.multinomial(total_inventory, [1/num_echelons]*num_echelons).astype(float)
    elif first_layer == "first_only":
        inventory = np.zeros(num_echelons, dtype=float)
        inventory[0] = ini_inventory * num_echelons
    else:
        raise ValueError("Invalid first_layer parameter.")

    balance = np.full(num_echelons, endowment, dtype=float)
    requests_count = np.zeros(num_echelons, dtype=int)

    if switch_random == 1:
        next_request_time = np.full(num_echelons, request_delay) + np.random.uniform(0, max_request_interval, num_echelons)
    else:
        next_request_time = np.full(num_echelons, request_delay) + (max_request_interval / 2)

    next_inventory_time = inventory_delay
    current_time = 0.0
    event_log = []

    for i in range(num_echelons):
        event_log.append({
            "Time Step": 0, "Agent": i + 1, "Requested": 0, "Collected": 0,
            "Completed": 0, "Inventory": inventory[i], "Earnings": 0,
            "Costs requested": 0, "Costs Inventory": 0, "Costs stockout": 0,
            "Net Change": 0, "Balance": balance[i],
            "Bankrupt": False
        })

    while current_time < time_end:
        next_event_time = min(next_inventory_time, next_request_time.min())
        if next_event_time > time_end:
            current_time = time_end
            break
        current_time = next_event_time

        if abs(current_time - next_inventory_time) < 1e-8:
            for i in range(num_echelons):
                cost = inventory[i] * price_inventory
                balance[i] -= cost
                event_log.append({
                    "Time Step": current_time, "Agent": i + 1, "Requested": 0, "Collected": 0,
                    "Completed": 0, "Inventory": inventory[i], "Earnings": 0,
                    "Costs requested": 0, "Costs Inventory": -cost, "Costs stockout": 0,
                    "Net Change": -cost, "Balance": balance[i],
                    "Bankrupt": (balance[i] < -(inventory[i] * price_sell) + price_request)
                })
            next_inventory_time += inventory_delay

        else:
            candidate = int(np.argmin(next_request_time))
            requests_count[candidate] += 1
            balance[candidate] -= price_request
            pred = candidate - 1 if candidate != 0 else num_echelons - 1

            if inventory[pred] > 0:
                inventory[candidate] += 1
                inventory[pred] -= 1
                balance[pred] += price_sell
                event_log.extend([
                    {
                        "Time Step": current_time, "Agent": candidate + 1, "Requested": 1, "Collected": 1,
                        "Completed": 0, "Inventory": inventory[candidate], "Earnings": 0,
                        "Costs requested": -price_request, "Costs Inventory": 0, "Costs stockout": 0,
                        "Net Change": -price_request, "Balance": balance[candidate],
                        "Bankrupt": (balance[candidate] < -(inventory[candidate] * price_sell) + price_request)
                    },
                    {
                        "Time Step": current_time, "Agent": pred + 1, "Requested": 0, "Collected": 0,
                        "Completed": 1, "Inventory": inventory[pred], "Earnings": price_sell,
                        "Costs requested": 0, "Costs Inventory": 0, "Costs stockout": 0,
                        "Net Change": price_sell, "Balance": balance[pred],
                        "Bankrupt": (balance[pred] < -(inventory[pred] * price_sell) + price_request)
                    }
                ])
            else:
                event_log.append({
                    "Time Step": current_time, "Agent": candidate + 1, "Requested": 1, "Collected": 0,
                    "Completed": 0, "Inventory": inventory[candidate], "Earnings": 0,
                    "Costs requested": -price_request, "Costs Inventory": 0, "Costs stockout": -STOCKOUT_COST,
                    "Net Change": -(price_request + STOCKOUT_COST), "Balance": balance[candidate],
                    "Bankrupt": (balance[candidate] < -(inventory[candidate] * price_sell) + price_request)
                })

            if switch_random == 1:
                next_request_time[candidate] = current_time + request_delay + np.random.uniform(0, max_request_interval)
            else:
                next_request_time[candidate] = current_time + request_delay + (max_request_interval / 2)

    return pd.DataFrame(event_log)


# =============================================================================
# MAIN LOOP — batch runner for supply chain sim with 10 RUNS per setting
# =============================================================================
if __name__ == '__main__':
    output_dir = "ZI_simulation_outputs_180_runs"
    os.makedirs(output_dir, exist_ok=True)

    first_layer_distributions = ["symmetrical", "random", "first_only"]
    inventory_costs = [0.05, 0.5, 1.0, 1.5]
    agent_counts = [4, 5, 6, 10, 15, 20, 40]
    initial_inventories = [1, 2, 3, 4, 5, 10, 20]

    for first_layer in first_layer_distributions:
        first_layer_dir = os.path.join(output_dir, f"FirstLayer_{first_layer}")
        os.makedirs(first_layer_dir, exist_ok=True)

        for inv_cost in inventory_costs:
            cost_dir = os.path.join(first_layer_dir, f"InventoryCost_{inv_cost}")
            os.makedirs(cost_dir, exist_ok=True)

            for num_agents in agent_counts:
                agents_dir = os.path.join(cost_dir, f"Agents_{num_agents}")
                os.makedirs(agents_dir, exist_ok=True)

                for ini_inv in initial_inventories:
                    print(f"\n=== Running simulations: Layer={first_layer}, Cost={inv_cost}, Agents={num_agents}, Inventory={ini_inv} ===")
                    all_runs = []

                    for run_id in range(10):
                        df = run_simulation(
                            sim_seed=run_id,
                            ini_inventory=ini_inv,
                            time_end=180,
                            num_echelons=num_agents,
                            price_inventory=inv_cost,
                            first_layer=first_layer
                        )
                        df["Run"] = run_id
                        all_runs.append(df)

                    combined_df = pd.concat(all_runs, ignore_index=True)
                    output_file = os.path.join(
                        agents_dir,
                        f"SimLog_ALLRUNS_Agents_{num_agents}_IniInventory_{ini_inv}_InvCost_{inv_cost}_180.csv"
                    )
                    combined_df.to_csv(output_file, index=False)
                    print(f"Saved combined CSV to: {output_file}")

    print("\nAll 10x batch simulations completed.")
    
    
    

############################################################################
# CALIBRATION — AVERAGED OVER RUNS
############################################################################

import os
import csv
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

# =============================================================================
# Parameter Grid
# =============================================================================
first_layer_distributions = ["symmetrical", "random", "first_only"]
inventory_costs = [0.05, 0.5, 1.0, 1.5]
agent_counts = [4, 5, 6, 10, 15, 20, 40]
initial_inventories = [1, 2, 3, 4, 5, 10, 20]

# =============================================================================
# Metrics Extraction for a Single Run
# =============================================================================
def extract_metrics_df(df, agent=1, block_size=6):
    inv_by_time = defaultdict(list)
    balances = {}
    bankrupt_flags = {}
    agent1_requests = 0
    agent1_collected = 0
    flux_clicks = []
    click_id = 0
    collected_click_ids = []

    for _, row in df.iterrows():
        try:
            time = float(row["Time Step"])
            if time < 0 or time > 180:
                continue

            agent_id = int(row["Agent"])
            inventory = float(row["Inventory"])
            balance = float(row["Balance"])
            bankrupt = str(row["Bankrupt"]).strip().lower() == "true"

            inv_by_time[time].append(inventory)
            balances[agent_id] = balance
            bankrupt_flags[agent_id] = bankrupt

            if agent_id == agent:
                if row["Requested"] == 1:
                    agent1_requests += 1
                    click_id += 1
                if row["Collected"] == 1:
                    agent1_collected += 1
                    collected_click_ids.append(click_id)
        except Exception:
            continue

    if len(collected_click_ids) >= block_size:
        for k in range(len(collected_click_ids) - block_size + 1):
            start = collected_click_ids[k]
            end = collected_click_ids[k + block_size - 1]
            flux_clicks.append(end - start)
        avg_inverse_flux = sum(flux_clicks) / len(flux_clicks)
        std_inverse_flux = pd.Series(flux_clicks).std()
    else:
        avg_inverse_flux = float('nan')
        std_inverse_flux = float('nan')

    var_sum = 0
    for invs in inv_by_time.values():
        mean_inv = sum(invs) / len(invs)
        var = sum((x - mean_inv) ** 2 for x in invs) / len(invs)
        var_sum += var
    avg_inv_variance = var_sum / len(inv_by_time) if inv_by_time else float('nan')

    final_balances = list(balances.values())
    mean_end_balance = sum(final_balances) / len(final_balances) if final_balances else float('nan')
    end_balance_variance = sum((b - mean_end_balance) ** 2 for b in final_balances) / len(final_balances) if final_balances else float('nan')

    bankrupt_count = sum(1 for v in bankrupt_flags.values() if v)
    bankruptcy_fraction = bankrupt_count / len(bankrupt_flags) if bankrupt_flags else float('nan')

    success_rate = agent1_collected / agent1_requests if agent1_requests > 0 else float('nan')

    return (
        avg_inverse_flux, std_inverse_flux, avg_inv_variance,
        mean_end_balance, end_balance_variance, bankruptcy_fraction, success_rate
    )

# =============================================================================
# Process One Simulation File (with all 10 runs)
# =============================================================================
def process_single_simulation(params, base_dir, output_file, lock):
    first_layer, inv_cost, agent_count, ini_inv = params
    folder = os.path.join(
        base_dir,
        f"FirstLayer_{first_layer}",
        f"InventoryCost_{inv_cost}",
        f"Agents_{agent_count}"
    )
    csv_name = f"SimLog_ALLRUNS_Agents_{agent_count}_IniInventory_{ini_inv}_InvCost_{inv_cost}_180.csv"
    csv_path = os.path.join(folder, csv_name)

    print(f"Processing: FL={first_layer}, IC={inv_cost}, A={agent_count}, IniInv={ini_inv}")

    try:
        if os.path.isfile(csv_path):
            df = pd.read_csv(csv_path)
            if 'Run' not in df.columns:
                print(f"[Skipping] No 'Run' column in: {csv_path}")
                return

            run_metrics = []
            for run_id, run_df in df.groupby("Run"):
                metrics = extract_metrics_df(run_df)
                run_metrics.append(metrics)

            avg_metrics = pd.DataFrame(run_metrics).mean().tolist()
            row = (first_layer, inv_cost, agent_count, ini_inv) + tuple(avg_metrics)

            with lock:
                with open(output_file, "a", newline='') as f_out:
                    writer = csv.writer(f_out)
                    writer.writerow(row)

            print(f" Written summary row for {csv_path}")

        else:
            print(f"[Missing] File not found: {csv_path}")
    except Exception as e:
        print(f"[Error] {csv_path}: {e}")

# =============================================================================
# Main Parallel Loop
# =============================================================================
def process_all_simulations_parallel(
    base_dir="ZI_simulation_outputs_180_runs",
    output_file="SimulationMetricsSummary_180_ALLRUNS.csv",
    max_workers=8
):
    parameter_grid = [
        (first_layer, inv_cost, agent_count, ini_inv)
        for first_layer in first_layer_distributions
        for inv_cost in inventory_costs
        for agent_count in agent_counts
        for ini_inv in initial_inventories
    ]

    with open(output_file, "w", newline='') as f_out:
        writer = csv.writer(f_out)
        writer.writerow([
            "FirstLayer", "InventoryCost", "Agents", "IniInventory",
            "AvgInverseFlux", "StdInverseFlux", "AvgInventoryVariance",
            "MeanEndBalance", "EndBalanceVariance",
            "BankruptcyFraction", "Agent1SuccessRate"
        ])

    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(lambda p: process_single_simulation(p, base_dir, output_file, lock), parameter_grid)

    print("\n All simulations processed and summarized (10-run average per combination).")

# =============================================================================
# Run It
# =============================================================================
if __name__ == "__main__":
    process_all_simulations_parallel()



########################################################################
# Descriptive Sats and Correlation Matrix
########################################################################


import pandas as pd
import numpy as np
from scipy.stats import spearmanr

# === Load your results file ===
summary_path = "SimulationMetricsSummary_180_ALLRUNS.csv"
df = pd.read_csv(summary_path)

# === Encode 'FirstLayer' as categorical numeric (for correlation) ===
df["FirstLayerCode"] = df["FirstLayer"].astype("category").cat.codes

# === Input factors and outcome metrics ===
input_factors = ["IniInventory", "InventoryCost", "Agents", "FirstLayerCode"]
outcome_metrics = [
    "AvgInverseFlux", "StdInverseFlux", "AvgInventoryVariance",
    "MeanEndBalance", "EndBalanceVariance", "BankruptcyFraction", "Agent1SuccessRate"
]

# Optional: rename for human-readable display
rename_inputs = {
    "IniInventory": "Inventory",
    "InventoryCost": "Inventory Cost",
    "Agents": "Number of Agents",
    "FirstLayerCode": "Initial Distribution"
}
rename_outcomes = {
    "AvgInverseFlux": "Average Inverse Flux",
    "StdInverseFlux": "Std Inverse Flux",
    "AvgInventoryVariance": "Inventory Variance",
    "MeanEndBalance": "End Balance",
    "EndBalanceVariance": "Balance Variance",
    "BankruptcyFraction": "Fraction Bankrupt",
    "Agent1SuccessRate": "Agent1 Success"
}

# === Descriptive Statistics Table ===
desc_stats = df[outcome_metrics].describe(percentiles=[.25, .5, .75]).T
desc_stats = desc_stats[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
desc_stats = desc_stats.rename(columns={
    "count": "Count", "mean": "Mean", "std": "Std", "min": "Min",
    "25%": "25%", "50%": "50%", "75%": "75%", "max": "Max"
})
desc_stats.rename(index=rename_outcomes, inplace=True)

# === Spearman Correlation Matrix ===
corr_matrix = pd.DataFrame(index=[rename_inputs[col] for col in input_factors],
                           columns=[rename_outcomes[col] for col in outcome_metrics])

for input_col in input_factors:
    for outcome_col in outcome_metrics:
        subset = df[[input_col, outcome_col]].dropna()
        rho, pval = spearmanr(subset[input_col], subset[outcome_col])
        if pval < 0.01:
            stars = "***"
        elif pval < 0.05:
            stars = "**"
        elif pval < 0.1:
            stars = "*"
        else:
            stars = ""
        corr_matrix.loc[rename_inputs[input_col], rename_outcomes[outcome_col]] = f"{rho:.2f}{stars}"

# === Display or export ===
print("\n=== Descriptive Statistics ===\n")
print(desc_stats.to_string())

print("\n=== Spearman Correlation Matrix (with significance) ===\n")
print(corr_matrix.to_string())

# Optional: save to CSV
desc_stats.to_csv("Descriptive_Stats.csv")
corr_matrix.to_csv("Spearman_Correlation_Matrix.csv")



##################################################################
# Agents Calibration
##################################################################

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

# === Matplotlib configuration for scientific style ===
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "axes.edgecolor": "black",
    "axes.linewidth": 1.0,
    "lines.linewidth": 1.5,
    "text.usetex": False
})

# === Load simulation summary data ===
df = pd.read_csv("SimulationMetricsSummary_180_ALLRUNS.csv")

# === Setup for plotting ===
distributions = ["symmetrical", "random", "first_only"]
dist_name_map = {
    "symmetrical": "Symmetric",
    "random": "Random",
    "first_only": "Asymmetric"
}

# === Plot a) — Std. Inverse Flux vs Agents at InventoryCost = 0.5 ===
unique_inventories = sorted(df["IniInventory"].unique())
n_inv = len(unique_inventories)
colors = sns.color_palette("Set2", n_colors=n_inv)

fig, axs = plt.subplots(1, 3, figsize=(14, 5.5), sharey=True)
fig.suptitle("a) Standard Deviation of Inverse Flow vs Number of Agents\nat Inventory Cost = 0.5", 
             fontsize=16, x=0.5, y=0.95)

for ax, dist in zip(axs, distributions):
    sub = df[(df["FirstLayer"] == dist) & (df["InventoryCost"] == 0.5)]

    for i, ini_inv in enumerate(unique_inventories):
        subset = sub[sub["IniInventory"] == ini_inv].sort_values("Agents")
        if subset.empty:
            continue
        ax.plot(subset["Agents"], subset["StdInverseFlux"], label=f"Avg Inv = {ini_inv}", color=colors[i])

    rho, pval = spearmanr(sub["Agents"], sub["StdInverseFlux"])
    ax.text(0.05, 0.95, f"Spearman’s ρ = {rho:.2f}\np = {pval:.4f}",
            transform=ax.transAxes, verticalalignment='top')

    ax.set_title(f"{dist_name_map[dist]} dist")
    ax.set_xlabel("Number of Agents")
    if ax == axs[0]:
        ax.set_ylabel("Std. of Inverse Flow")
    
    ax.set_xticks(sorted(sub["Agents"].unique()))
    ax.grid(True)

axs[-1].legend(title="Average Initial Inventory", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("a_std_inverse_flow_vs_agents_fixed_cost.png", dpi=600, bbox_inches='tight')
plt.show()

# === Plot b) — Mean End Balance vs Inventory Cost at Avg Initial Inventory = 2 ===
fig, axs = plt.subplots(1, 3, figsize=(14, 5.5), sharey=True)
fig.suptitle("b) Mean End Balance vs Inventory Cost\nat Average Initial Inventory = 2", 
             fontsize=16, x=0.5, y=0.95)

unique_agents = sorted(df["Agents"].unique())
n_agents = len(unique_agents)
agent_colors = sns.color_palette("Set2", n_colors=n_agents)

for ax, dist in zip(axs, distributions):
    sub = df[(df["FirstLayer"] == dist) & (df["IniInventory"] == 2)]

    for i, agent_count in enumerate(unique_agents):
        subset = sub[sub["Agents"] == agent_count].sort_values("InventoryCost")
        if subset.empty:
            continue
        ax.plot(subset["InventoryCost"], subset["MeanEndBalance"], label=f"Agents {agent_count}", color=agent_colors[i])

    rho, pval = spearmanr(sub["InventoryCost"], sub["MeanEndBalance"])
    ax.text(0.05, 0.95, f"Spearman’s ρ = {rho:.2f}\np = {pval:.4f}",
            transform=ax.transAxes, verticalalignment='top')

    ax.set_title(f"{dist_name_map[dist]} dist")
    ax.set_xlabel("Inventory Cost")
    if ax == axs[0]:
        ax.set_ylabel("Mean End Balance")
    
    ax.set_xticks(sorted(sub["InventoryCost"].unique()))
    ax.grid(True)

axs[-1].legend(title="Agent Count", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("b_mean_end_balance_vs_cost_avg_inventory_2.png", dpi=600, bbox_inches='tight')
plt.show()


##################################################################
# Inventory Calibration
##################################################################

# === Load data ===
df = pd.read_csv("SimulationMetricsSummary_180_ALLRUNS.csv")

# === Setup ===
unique_inventories = sorted(df["IniInventory"].unique())
unique_agents = sorted(df["Agents"].unique())
unique_costs = sorted(df["InventoryCost"].unique())

# Use consistent colors
inv_colors = sns.color_palette("Set2", n_colors=len(unique_inventories))
agent_colors = sns.color_palette("Set2", n_colors=len(unique_agents))

# === Panel a) — Std. Inverse Flux vs Avg Initial Inventory @ InventoryCost = 0.5 ===
fig, axs = plt.subplots(1, 3, figsize=(14, 5.5), sharey=True)
fig.suptitle("a) Standard Deviation of Inverse Flow vs Average Initial Inventory\nat Inventory Cost = 0.5", 
             fontsize=16, x=0.5, y=0.95)

for ax, dist in zip(axs, distributions):
    sub = df[(df["FirstLayer"] == dist) & (df["InventoryCost"] == 0.5)]

    for i, agent_count in enumerate(unique_agents):
        subset = sub[sub["Agents"] == agent_count].sort_values("IniInventory")
        if subset.empty:
            continue
        ax.plot(subset["IniInventory"], subset["StdInverseFlux"], label=f"Agents {agent_count}", color=agent_colors[i])

    rho, pval = spearmanr(sub["IniInventory"], sub["StdInverseFlux"])
    ax.text(0.05, 0.95, f"Spearman’s ρ = {rho:.2f}\np = {pval:.4f}",
            transform=ax.transAxes, verticalalignment='top')

    ax.set_title(f"{dist_name_map[dist]} dist")
    ax.set_xlabel("Average Initial Inventory")
    if ax == axs[0]:
        ax.set_ylabel("Std. of Inverse Flow")

    ax.set_xticks(unique_inventories)
    ax.grid(True)

axs[-1].legend(title="Agent Count", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("a_std_inverse_flow_vs_avg_inventory_fixed_cost.png", dpi=600, bbox_inches='tight')
plt.show()

# === Panel b) — LOG–LOG Std. Inverse Flux vs Avg Initial Inventory ===
fig, axs = plt.subplots(1, 3, figsize=(14, 5.5), sharey=True)
fig.suptitle("b) Log–Log: Standard Deviation of Inverse Flow vs Average Initial Inventory\nat Inventory Cost = 0.5", 
             fontsize=16, x=0.5, y=0.95)

for ax, dist in zip(axs, distributions):
    sub = df[(df["FirstLayer"] == dist) & (df["InventoryCost"] == 0.5)]

    for i, agent_count in enumerate(unique_agents):
        subset = sub[sub["Agents"] == agent_count].sort_values("IniInventory")
        if subset.empty:
            continue
        ax.plot(subset["IniInventory"], subset["StdInverseFlux"], label=f"Agents {agent_count}", color=agent_colors[i])

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_title(f"{dist_name_map[dist]} dist")
    ax.set_xlabel("Average Initial Inventory (log scale)")
    if ax == axs[0]:
        ax.set_ylabel("Std. of Inverse Flow (log scale)")

    ax.grid(True, which="both", linestyle="--", linewidth=0.6)

axs[-1].legend(title="Agent Count", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("b_loglog_std_inverse_flow_vs_avg_inventory_fixed_cost.png", dpi=600, bbox_inches='tight')
plt.show()

# === Panel c) — Mean End Balance vs Inventory Cost @ Agents = 5 ===
fig, axs = plt.subplots(1, 3, figsize=(14, 5.5), sharey=True)
fig.suptitle("b) Mean End Balance vs Inventory Cost\nat Agent Count = 5", 
             fontsize=16, x=0.5, y=0.95)

for ax, dist in zip(axs, distributions):
    sub = df[(df["FirstLayer"] == dist) & (df["Agents"] == 5)]

    for i, ini_inv in enumerate(unique_inventories):
        subset = sub[sub["IniInventory"] == ini_inv].sort_values("InventoryCost")
        if subset.empty:
            continue
        ax.plot(subset["InventoryCost"], subset["MeanEndBalance"], label=f"Avg Inv = {ini_inv}", color=inv_colors[i])

    ax.set_title(f"{dist_name_map[dist]} dist")
    ax.set_xlabel("Inventory Cost")
    if ax == axs[0]:
        ax.set_ylabel("Mean End Balance")

    ax.set_xticks(unique_costs)
    ax.grid(True)

axs[-1].legend(title="Average Initial Inventory", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("c_mean_end_balance_vs_cost_fixed_agents_5.png", dpi=600, bbox_inches='tight')
plt.show()





# =============================================================================
##############################################################################
# Analisis of System Fragility
##############################################################################
# =============================================================================

# -------------------------------------------------------------------------
# Zero-Intelligence SIMULATION
# -------------------------------------------------------------------------

##############################################################################
# Zero Intelligence Simulation for FRACTIONAL Inventory
##############################################################################


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import floor
import os
import csv
import concurrent.futures
import warnings
from scipy.stats import linregress

# =============================================================================
# Suppress divide‐by‐zero warnings from NumPy
# =============================================================================
warnings.filterwarnings("ignore", category=RuntimeWarning)
np.seterr(divide='ignore', invalid='ignore')


# =============================================================================
# Helper Function: Run a Single Simulation (Streaming to CSV)
# =============================================================================
def run_simulation_to_csv(sim_seed,
                          ini_inventory,
                          output_csv_path,
                          plot_needed=False,
                          plot_path=None,
                          time_end=600_000,      # default high for streaming
                          switch_random=1,
                          num_echelons=10,
                          inventory_delay=1.0,
                          request_delay=0.01,
                          max_request_interval=4.0,
                          endowment=30.0,
                          price_sell=10.0,
                          price_request=0.0,
                          price_inventory=2.0,
                          STOCKOUT_COST=0.0,
                          first_layer="symmetrical"):
    """
    Runs one simulation with the specified parameters and writes each event directly
    to a CSV file at `output_csv_path`. If plot_needed=True (and num_echelons==5),
    accumulates only events up to 600s for in‐memory plotting, then saves PDF to plot_path.
    """
    np.random.seed(sim_seed)  # Set unique random seed for reproducibility

    # Initialize inventory according to first_layer policy
    if first_layer == "symmetrical":
        inventory = np.full(num_echelons, ini_inventory, dtype=float)
    elif first_layer == "random":
        total_inventory = ini_inventory * num_echelons
        inventory = np.random.multinomial(total_inventory, [1 / num_echelons] * num_echelons).astype(float)
    elif first_layer == "first_only":
        inventory = np.zeros(num_echelons, dtype=float)
        inventory[0] = ini_inventory * num_echelons
    else:
        raise ValueError("Invalid first_layer parameter. Choose 'symmetrical', 'random', or 'first_only'.")

    balance = np.full(num_echelons, endowment, dtype=float)
    requests_count = np.zeros(num_echelons, dtype=int)

    # Initialize next_request_time array
    if switch_random == 1:
        next_request_time = np.full(num_echelons, request_delay) + np.random.uniform(0, max_request_interval, num_echelons)
    else:
        next_request_time = np.full(num_echelons, request_delay) + (max_request_interval / 2.0)

    next_inventory_time = inventory_delay
    current_time = 0.0

    # Prepare CSV writer
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    csv_file = open(output_csv_path, mode='w', newline='')
    fieldnames = [
        "Time Step", "Agent", "Requested", "Collected", "Completed",
        "Inventory", "Earnings", "Costs requested", "Costs Inventory",
        "Costs stockout", "Net Change", "Balance", "Bankrupt"
    ]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    # If plotting is needed, accumulate only events with time <= 600
    if plot_needed:
        plot_events = []

    # WRITE INITIAL STATE FOR EACH AGENT
    for i in range(num_echelons):
        row = {
            "Time Step": 0.0,
            "Agent": i + 1,
            "Requested": 0,
            "Collected": 0,
            "Completed": 0,
            "Inventory": inventory[i],
            "Earnings": 0.0,
            "Costs requested": 0.0,
            "Costs Inventory": 0.0,
            "Costs stockout": 0.0,
            "Net Change": 0.0,
            "Balance": balance[i],
            "Bankrupt": False
        }
        writer.writerow(row)
        if plot_needed and 0.0 <= 600.0:
            plot_events.append(row)

    # MAIN EVENT LOOP (streaming each event)
    while current_time < time_end:
        next_event_time = min(next_inventory_time, next_request_time.min())
        if next_event_time > time_end:
            current_time = time_end
            break
        current_time = next_event_time

        # INVENTORY COST EVENT
        if abs(current_time - next_inventory_time) < 1e-8:
            for i in range(num_echelons):
                cost = inventory[i] * price_inventory
                balance[i] -= cost
                row = {
                    "Time Step": current_time,
                    "Agent": i + 1,
                    "Requested": 0,
                    "Collected": 0,
                    "Completed": 0,
                    "Inventory": inventory[i],
                    "Earnings": 0.0,
                    "Costs requested": 0.0,
                    "Costs Inventory": -cost,
                    "Costs stockout": 0.0,
                    "Net Change": -cost,
                    "Balance": balance[i],
                    "Bankrupt": (balance[i] < -(inventory[i] * price_sell) + price_request)
                }
                writer.writerow(row)
                if plot_needed and current_time <= 600.0:
                    plot_events.append(row)
            next_inventory_time += inventory_delay

        # REQUEST EVENT
        else:
            candidate = int(np.argmin(next_request_time))
            requests_count[candidate] += 1
            balance[candidate] -= price_request

            pred = candidate - 1 if candidate != 0 else num_echelons - 1

            # Successful fulfillment
            if inventory[pred] > 0:
                inventory[candidate] += 1
                inventory[pred] -= 1
                balance[pred] += price_sell

                row_cand = {
                    "Time Step": current_time,
                    "Agent": candidate + 1,
                    "Requested": 1,
                    "Collected": 1,
                    "Completed": 0,
                    "Inventory": inventory[candidate],
                    "Earnings": 0.0,
                    "Costs requested": -price_request,
                    "Costs Inventory": 0.0,
                    "Costs stockout": 0.0,
                    "Net Change": -price_request,
                    "Balance": balance[candidate],
                    "Bankrupt": (balance[candidate] < -(inventory[candidate] * price_sell) + price_request)
                }
                writer.writerow(row_cand)
                if plot_needed and current_time <= 600.0:
                    plot_events.append(row_cand)

                row_pred = {
                    "Time Step": current_time,
                    "Agent": pred + 1,
                    "Requested": 0,
                    "Collected": 0,
                    "Completed": 1,
                    "Inventory": inventory[pred],
                    "Earnings": price_sell,
                    "Costs requested": 0.0,
                    "Costs Inventory": 0.0,
                    "Costs stockout": 0.0,
                    "Net Change": price_sell,
                    "Balance": balance[pred],
                    "Bankrupt": (balance[pred] < -(inventory[pred] * price_sell) + price_request)
                }
                writer.writerow(row_pred)
                if plot_needed and current_time <= 600.0:
                    plot_events.append(row_pred)

            # Stockout scenario
            else:
                row_so = {
                    "Time Step": current_time,
                    "Agent": candidate + 1,
                    "Requested": 1,
                    "Collected": 0,
                    "Completed": 0,
                    "Inventory": inventory[candidate],
                    "Earnings": 0.0,
                    "Costs requested": -price_request,
                    "Costs Inventory": 0.0,
                    "Costs stockout": -STOCKOUT_COST,
                    "Net Change": -(price_request + STOCKOUT_COST),
                    "Balance": balance[candidate],
                    "Bankrupt": (balance[candidate] < -(inventory[candidate] * price_sell) + price_request)
                }
                writer.writerow(row_so)
                if plot_needed and current_time <= 600.0:
                    plot_events.append(row_so)

            # Schedule next request for this agent
            if switch_random == 1:
                next_request_time[candidate] = current_time + request_delay + np.random.uniform(0, max_request_interval)
            else:
                next_request_time[candidate] = current_time + request_delay + (max_request_interval / 2.0)

    # CLEAN UP CSV FILE
    csv_file.close()

    # IF PLOTTING IS NEEDED, BUILD DATAFRAME AND PLOT
    if plot_needed and plot_path is not None:
        df_plot = pd.DataFrame(plot_events)
        plot_event_log_multisubplot(df_plot, save_path=plot_path)


# =============================================================================
# Helper Function: Plot Event Log Per Agent (Multi‐Subplot) WITHOUT Showing
# =============================================================================
def plot_event_log_multisubplot(event_log_df, save_path=None):
    participants = sorted(event_log_df["Agent"].unique())
    num_participants = len(participants)
    fig, axs = plt.subplots(num_participants, 1, figsize=(11.7, 8.3), sharex=True)
    if num_participants == 1:
        axs = [axs]

    max_inv = event_log_df["Inventory"].max()
    max_bal = event_log_df["Balance"].max()
    scale_factor = max_inv / max_bal if max_bal != 0 else 1

    for ax, agent_id in zip(axs, participants):
        part_data = event_log_df[event_log_df["Agent"] == agent_id].sort_values("Time Step")
        ax.plot(part_data["Time Step"], part_data["Inventory"], color="blue", label="Inventory")
        ax.plot(part_data["Time Step"], part_data["Balance"] * scale_factor,
                color="orange", label="Balance")

        # Highlight bankrupt periods
        bankrupt_mask = part_data["Bankrupt"].values.astype(bool)
        times = part_data["Time Step"].values
        bal_scaled = (part_data["Balance"] * scale_factor).values

        segments = []
        current_seg_times = []
        current_seg_balance = []
        for i in range(len(bankrupt_mask)):
            if bankrupt_mask[i]:
                current_seg_times.append(times[i])
                current_seg_balance.append(bal_scaled[i])
            else:
                if current_seg_times:
                    segments.append((current_seg_times, current_seg_balance))
                    current_seg_times = []
                    current_seg_balance = []
        if current_seg_times:
            segments.append((current_seg_times, current_seg_balance))

        bankrupt_label_used = False
        for seg_times, seg_balance in segments:
            if len(seg_times) == 1:
                seg_times = [seg_times[0], seg_times[0] + 1e-6]
                seg_balance = [seg_balance[0], seg_balance[0]]
            if not bankrupt_label_used:
                ax.plot(seg_times, seg_balance, color="red", linewidth=2, label="Bankrupt")
                bankrupt_label_used = True
            else:
                ax.plot(seg_times, seg_balance, color="red", linewidth=2)

        ax.set_title(f"Agent {agent_id}")
        ax.set_ylabel("Inventory")
        secax = ax.secondary_yaxis('right', functions=(lambda x: x / scale_factor, lambda x: x * scale_factor))
        secax.set_ylabel("Balance")

    handles, labels = axs[0].get_legend_handles_labels()
    unique = {}
    for handle, label in zip(handles, labels):
        if label not in unique:
            unique[label] = handle
    axs[0].legend(unique.values(), unique.keys(), loc="upper left")
    plt.xlabel("Time (s)")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, format="pdf")
    plt.close(fig)


# =============================================================================
# Worker Function: Called in Parallel
# =============================================================================
def worker_task(params):
    """
    params: (first_layer, inv_cost, num_agents, ini_inv)
    Creates its own output directory, runs the simulation (streaming),
    and—if needed—creates the 600s plot.
    """
    first_layer, inv_cost, num_agents, ini_inv = params

    # Construct output directories
    base_dir = "ZI_simulation_outputs"
    first_layer_dir = os.path.join(base_dir, f"FirstLayer_{first_layer}")
    cost_dir = os.path.join(first_layer_dir, f"InventoryCost_{inv_cost}")
    agents_dir = os.path.join(cost_dir, f"Agents_{num_agents}")
    os.makedirs(agents_dir, exist_ok=True)

    # Paths for CSV and (optionally) PDF plot
    csv_filename = os.path.join(
        agents_dir,
        f"SimLog_Agents_{num_agents}_IniInventory_{ini_inv}_InvCost_{inv_cost}.csv"
    )

    # Determine if we need to plot the first 600s (only when num_agents == 5)
    plot_needed = (num_agents == 5)
    if plot_needed:
        plot_filename = os.path.join(
            agents_dir,
            f"EventLogPlot600s_Agents_{num_agents}_IniInventory_{ini_inv}_InvCost_{inv_cost}.pdf"
        )
    else:
        plot_filename = None

    # Run the simulation, streaming to CSV
    print(f"Starting simulation: FirstLayer={first_layer}, InventoryCost={inv_cost}, Agents={num_agents}, IniInventory={ini_inv}")
    run_simulation_to_csv(
        sim_seed=55,
        ini_inventory=ini_inv,
        output_csv_path=csv_filename,
        plot_needed=plot_needed,
        plot_path=plot_filename,
        time_end=1_000_000,         # run full 1 million seconds
        num_echelons=num_agents,
        price_inventory=inv_cost,
        first_layer=first_layer
    )
    print(f"Finished simulation: CSV -> {csv_filename}")
    if plot_needed:
        print(f"600s plot saved -> {plot_filename}")


# =============================================================================
# MAIN — Parallel Batch Runner
# =============================================================================
if __name__ == "__main__":

    first_layer_distributions = ["first_only"]
    inventory_costs = [0.5]
    agent_counts = [5]
    initial_inventories = [0.2, 0.4, 0.6, 0.8]

    all_tasks = []
    for first_layer in first_layer_distributions:
        for inv_cost in inventory_costs:
            for num_agents in agent_counts:
                for ini_inv in initial_inventories:
                    all_tasks.append((first_layer, inv_cost, num_agents, ini_inv))

    # Run sequentially (simple for loop — safe everywhere!)
    for params in all_tasks:
        print(f"\n=== Running simulation for params: {params} ===")
        worker_task(params)

    print("\n All custom asymmetric runs completed.")







##############################################################################
# Absolute Time
##############################################################################

# ---------------------------------------------------------------------------
# Inverse flux of 5 items measured in TIME: actual flux times (so collected == 1)
# ---------------------------------------------------------------------------

import os
import csv
import numpy as np

def extract_flux_times(csv_path, agent=1, block=6):
    times = []

    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["Agent"]) == agent and row["Collected"] == '1':
                times.append(float(row["Time Step"]))

    flux = []
    if len(times) >= block:
        for k in range(len(times) - block + 1):
            start = times[k]
            end = times[k + block - 1]
            flux.append(end - start)

    # Add debug / sanity check
    print(f"Extracted {len(times)} collected events for agent {agent}. Computed {len(flux)} flux times (block size={block}).")

    return flux


if __name__ == "__main__":

    base_dir = "ZI_simulation_outputs"
    agent = 1
    inv_cost = 0.5
    num_agents = 5
    block_size = 6

    # Group 1 — symmetrical
    first_layer = "symmetrical"
    initial_inventories_sym = [1, 2, 3, 4, 5, 10, 20, 100, 1000]

    # Group 2 — first_only
    first_layer_first = "first_only"
    initial_inventories_first = [0.2, 0.4, 0.6, 0.8]

    # Container for ALL flux results
    all_flux_rows = []

    # Process symmetrical
    for ini_inv in initial_inventories_sym:
        folder = os.path.join(base_dir, f"FirstLayer_{first_layer}", f"InventoryCost_{inv_cost}", f"Agents_{num_agents}")
        csv_filename = f"SimLog_Agents_{num_agents}_IniInventory_{ini_inv}_InvCost_{inv_cost}.csv"
        csv_path = os.path.join(folder, csv_filename)

        flux_times = extract_flux_times(csv_path, agent=agent, block=block_size)

        for i, ft in enumerate(flux_times, start=1):
            all_flux_rows.append([ini_inv, i, ft])

        print(f"Extracted flux times for symmetrical IniInventory={ini_inv}")

    # Process first_only
    for ini_inv in initial_inventories_first:
        folder = os.path.join(base_dir, f"FirstLayer_{first_layer_first}", f"InventoryCost_{inv_cost}", f"Agents_{num_agents}")
        csv_filename = f"SimLog_Agents_{num_agents}_IniInventory_{ini_inv}_InvCost_{inv_cost}.csv"
        csv_path = os.path.join(folder, csv_filename)

        flux_times = extract_flux_times(csv_path, agent=agent, block=block_size)

        for i, ft in enumerate(flux_times, start=1):
            all_flux_rows.append([ini_inv, i, ft])

        print(f"Extracted flux times for first_only IniInventory={ini_inv}")

    # Save the combined CSV
    out_csv_path = os.path.join(base_dir, "Flux_Agent1.csv")
    with open(out_csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["ini_inventory", "flux_id", "flux_time"])
        for row in all_flux_rows:
            w.writerow(row)

    print(f"\n All flux times saved to {out_csv_path}")


# ---------------------------------------------------------------------------
# Inverse flux of 5 items measured in TIME: SCHEDULE (so requested == 1)
# ---------------------------------------------------------------------------

import os
import csv

# === Function to extract request times ===
def extract_request_times(csv_path, agent=1, block=6):
    times = []

    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["Agent"]) == agent and row["Requested"] == '1':
                times.append(float(row["Time Step"]))

    flux = []
    if len(times) >= block:
        for k in range(len(times) - block + 1):
            start = times[k]
            end = times[k + block - 1]
            flux.append(end - start)

    print(f"Extracted {len(times)} requested events for agent {agent}. Computed {len(flux)} schedule times (block size={block}).")

    return flux

# === Main ===
if __name__ == "__main__":

    base_dir = "ZI_simulation_outputs"
    agent = 1
    inv_cost = 0.5
    num_agents = 5
    block_size = 6

    first_layer = "symmetrical"
    ini_inv = 1  # ONLY inventory = 1

    # Path to the relevant CSV
    folder = os.path.join(base_dir, f"FirstLayer_{first_layer}", f"InventoryCost_{inv_cost}", f"Agents_{num_agents}")
    csv_filename = f"SimLog_Agents_{num_agents}_IniInventory_{ini_inv}_InvCost_{inv_cost}.csv"
    csv_path = os.path.join(folder, csv_filename)

    # Extract request times
    request_times = extract_request_times(csv_path, agent=agent, block=block_size)

    # Save schedule file
    out_csv_path = os.path.join(base_dir, "Schedule_File_Time.csv")
    with open(out_csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["flux_id", "flux_time"])  # Same structure as Flux_Agent1.csv but only with flux_id and flux_time
        for i, ft in enumerate(request_times, start=1):
            w.writerow([i, ft])

    print(f"\nSchedule file saved to {out_csv_path}")





# ---------------------------------------------------------------------------
# Inverse flux of 5 items: plot of NON-sorted inverse flow times for 0.2, 0.6, 2, 1000 and schedule
# ---------------------------------------------------------------------------

import os
import pandas as pd
import matplotlib.pyplot as plt

# === Matplotlib configuration for scientific style ===
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "axes.edgecolor": "black",
    "axes.linewidth": 1.0,
    "lines.linewidth": 1.5,
    "text.usetex": False
})

# === Configuration ===
base_dir = "ZI_simulation_outputs"
flux_csv_path = os.path.join(base_dir, "Flux_Agent1.csv")
schedule_csv_path = os.path.join(base_dir, "Schedule_File_Time.csv")  # NEW

invs_to_plot = [0.2, 0.6, 2, 1000]
flux_lo = 50
flux_hi = 150

marker_styles = ['^', 's', 'x', 'o']
grey_colors = ['#000000', '#333333', '#666666', '#999999']

# === Load flux CSV ===
df = pd.read_csv(flux_csv_path)

# === Plot preparation ===
fig, ax = plt.subplots(figsize=(7, 5))

# First plot inventory curves
for idx, ini_inv in enumerate(invs_to_plot):
    sub = df[(df['ini_inventory'] == ini_inv) &
             (df['flux_id'] >= flux_lo) &
             (df['flux_id'] <= flux_hi)]

    x_values = sub['flux_id']
    loop_times = sub['flux_time']
    label = f"IniInv={ini_inv}"

    ax.plot(
        x_values,
        loop_times,
        label=label,
        color=grey_colors[idx % len(grey_colors)],
        marker=marker_styles[idx % len(marker_styles)],
        markersize=4,
        markeredgewidth=0.8,
        markerfacecolor='white',
        markeredgecolor=grey_colors[idx % len(grey_colors)],
        linewidth=1.5,
        linestyle='-'
    )

# === Add request schedule line ===
schedule_df = pd.read_csv(schedule_csv_path)

# Select same flux_id range
schedule_sub = schedule_df[(schedule_df['flux_id'] >= flux_lo) & (schedule_df['flux_id'] <= flux_hi)]

ax.plot(
    schedule_sub['flux_id'],
    schedule_sub['flux_time'],
    label="Request Schedule",
    color='black',
    linestyle='--',
    linewidth=0.8,          
    dashes=(6, 4)            
)


# === Finalize plot ===
ax.set_xlabel(f"Loop Index ({flux_lo} to {flux_hi})", fontsize=12)
ax.set_ylabel("Inverse Flow Time [seconds]", fontsize=12)
ax.set_title(f"Inverse Flow Times of Selected Configurations (Index {flux_lo}–{flux_hi})", fontsize=14)
ax.grid(True)
ax.legend(frameon=True, edgecolor='black')
fig.tight_layout()

# === Save combined plot ===
combined_plot_path = os.path.join(base_dir, "Combined_Flux_Plot_Agent1_Monochrome.png")
plt.savefig(combined_plot_path, dpi=150)
plt.close()

print(f"Saved combined plot: {combined_plot_path}")




# ---------------------------------------------------------------------------
# MEAN AND STDEV of Inverse Flow Times [seconds]
# ---------------------------------------------------------------------------


import os
import pandas as pd

# === Configuration ===
base_dir = "ZI_simulation_outputs"
flux_csv_path = os.path.join(base_dir, "Flux_Agent1.csv")

# === Inventory groups ===
initial_inventories_sym = [1, 2, 3, 4, 5, 10, 20, 100, 1000]
initial_inventories_first = [0.2, 0.4, 0.6, 0.8]

# === All inventories ===
all_invs = initial_inventories_sym + initial_inventories_first

# === Load flux CSV ===
df = pd.read_csv(flux_csv_path)

# === Compute mean and std per inventory ===
summary_rows = []

for ini_inv in all_invs:
    sub = df[df['ini_inventory'] == ini_inv]
    mean_flux = sub['flux_time'].mean()
    std_flux = sub['flux_time'].std()

    summary_rows.append({
        "Inventory": ini_inv,
        "Mean Flux Time": mean_flux,
        "Std Dev Flux Time": std_flux
    })

# === Build summary DataFrame ===
summary_df = pd.DataFrame(summary_rows).sort_values(by='Inventory')

# === Save single summary CSV ===
summary_csv_path = os.path.join(base_dir, "flux_summary_per_inventory.csv")
summary_df.to_csv(summary_csv_path, index=False)

# === Print summary to console ===
print("\nFlux Summary Per Inventory:\n")
print(summary_df.to_string(index=False))



# ---------------------------------------------------------------------------
# Plot MEAN AND STDEV of Inverse Flow Times [seconds] as function of inventory
# ---------------------------------------------------------------------------

import os
import pandas as pd
import matplotlib.pyplot as plt

# === Matplotlib configuration for scientific style ===
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.edgecolor": "black",
    "axes.linewidth": 1.0,
    "lines.linewidth": 1.5,
    "text.usetex": False
})

# === Load summary CSV ===
base_dir = "ZI_simulation_outputs"
summary_csv_path = os.path.join(base_dir, "flux_summary_per_inventory.csv")
df = pd.read_csv(summary_csv_path)

# === Sort by inventory for plotting ===
df = df.sort_values(by="Inventory")

# === Grey color ===
grey_color = "#444444"
marker_style = 'o'

# === Create 2x2 panel ===
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# Panel a) — Mean Flux Time, Linear
axs[0, 0].plot(df['Inventory'], df['Mean Flux Time'],
               color=grey_color,
               marker=marker_style,
               markersize=5,
               markerfacecolor='white',
               markeredgewidth=1.0,
               markeredgecolor=grey_color,
               linestyle='-')
axs[0, 0].set_xlabel("Initial Inventory")
axs[0, 0].set_ylabel("Mean Inverse Flow Time [seconds]")
axs[0, 0].grid(True)
axs[0, 0].set_title("a) Mean Inverse Flow Time\nvs. Initial Inventory")

# Panel b) — Std Dev Flux Time, Linear
axs[0, 1].plot(df['Inventory'], df['Std Dev Flux Time'],
               color=grey_color,
               marker=marker_style,
               markersize=5,
               markerfacecolor='white',
               markeredgewidth=1.0,
               markeredgecolor=grey_color,
               linestyle='-')
axs[0, 1].set_xlabel("Initial Inventory")
axs[0, 1].set_ylabel("StdDev of Inverse Flow Time [seconds]")
axs[0, 1].grid(True)
axs[0, 1].set_title("b) StdDev of Inverse Flow Time\nvs. Initial Inventory")

# Panel c) — Mean Flux Time, Log-Log
axs[1, 0].plot(df['Inventory'], df['Mean Flux Time'],
               color=grey_color,
               marker=marker_style,
               markersize=5,
               markerfacecolor='white',
               markeredgewidth=1.0,
               markeredgecolor=grey_color,
               linestyle='-')
axs[1, 0].set_xlabel("Initial Inventory")
axs[1, 0].set_ylabel("Mean Inverse Flow Time [seconds]")
axs[1, 0].set_xscale('log')
axs[1, 0].set_yscale('log')
axs[1, 0].grid(True, which='both', linestyle='--', linewidth=0.5)
axs[1, 0].set_title("c) Mean Inverse Flow Time\nvs. Initial Inventory (log-log)")

# Panel d) — Std Dev Flux Time, Log-Log
axs[1, 1].plot(df['Inventory'], df['Std Dev Flux Time'],
               color=grey_color,
               marker=marker_style,
               markersize=5,
               markerfacecolor='white',
               markeredgewidth=1.0,
               markeredgecolor=grey_color,
               linestyle='-')
axs[1, 1].set_xlabel("Initial Inventory")
axs[1, 1].set_ylabel("StdDev of Inverse Flow Time [seconds]")
axs[1, 1].set_xscale('log')
axs[1, 1].set_yscale('log')
axs[1, 1].grid(True, which='both', linestyle='--', linewidth=0.5)
axs[1, 1].set_title("d) StdDev of Inverse Flow Time\nvs. Initial Inventory (log-log)")

# === Final layout ===
plt.tight_layout()

# === Save figure ===
# High quality PNG
png_path = os.path.join(base_dir, "FluxSummary_Panels.png")
plt.savefig(png_path, dpi=300)  # High dpi for thesis
print(f"\nSaved high-quality PNG: {png_path}")

# Optional: save also as PDF (perfect for LaTeX import)
pdf_path = os.path.join(base_dir, "FluxSummary_Panels.pdf")
plt.savefig(pdf_path)
print(f"Saved PDF: {pdf_path}")

# Close the figure
plt.close()



# ---------------------------------------------------------------------------
# Plot sorted inverse flow times + schedule measured in absolute TIME linear
# ---------------------------------------------------------------------------


import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === Matplotlib configuration for scientific style ===
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "axes.edgecolor": "black",
    "axes.linewidth": 1.0,
    "lines.linewidth": 1.5,
    "text.usetex": False
})

# === Configuration ===
base_dir = "ZI_simulation_outputs"
flux_csv_path = os.path.join(base_dir, "Flux_Agent1.csv")
schedule_csv_path = os.path.join(base_dir, "Schedule_File_Time.csv")

# === Load data ===
df = pd.read_csv(flux_csv_path)
schedule_df = pd.read_csv(schedule_csv_path)

# === Inventory list ===
invs_to_plot = sorted(df['ini_inventory'].unique())

# === Color palette ===
# Use Tableau tab20 → 20 colors, good for many curves
from matplotlib.cm import get_cmap
cmap = get_cmap("tab20")
colors = [cmap(i) for i in range(len(invs_to_plot))]

# === Plot preparation ===
fig, ax = plt.subplots(figsize=(8, 6))

# Plot each inventory — dots only
for idx, ini_inv in enumerate(invs_to_plot):
    sub = df[df['ini_inventory'] == ini_inv]['flux_time']
    sorted_times = np.sort(sub.values)

    x_values = np.arange(1, len(sorted_times) + 1)
    label = f"IniInv={ini_inv}"

    ax.plot(
        x_values,
        sorted_times,
        label=label,
        color=colors[idx % len(colors)],
        marker='o',
        linestyle='None',  # no connecting line
        markersize=2       # small dot
    )

# === Plot request schedule (sorted) — keep as dashed black line
sorted_schedule_times = np.sort(schedule_df['flux_time'].values)
x_schedule = np.arange(1, len(sorted_schedule_times) + 1)

ax.plot(
    x_schedule,
    sorted_schedule_times,
    label="Request Schedule",
    color='black',
    linestyle='--',
    linewidth=0.8,
    dashes=(6, 4)
)

# === Finalize plot ===
ax.set_xlabel("Index of Sorted Inverse Flow Time", fontsize=12)
ax.set_ylabel("Inverse Flow Time [seconds]", fontsize=12)
ax.set_title("Sorted Inverse Flow Times", fontsize=14)
ax.grid(True)
ax.legend(frameon=True, edgecolor='black', loc='best')
fig.tight_layout()

# === Save figure ===
plot_path_png = os.path.join(base_dir, "Sorted_Inverse_Flow_Times_Dots.png")
plot_path_pdf = os.path.join(base_dir, "Sorted_Inverse_Flow_Times_Dots.pdf")

plt.savefig(plot_path_png, dpi=300)
plt.savefig(plot_path_pdf)

plt.close()

print(f"\nSaved plot as PNG: {plot_path_png}")
print(f"Saved plot as PDF: {plot_path_pdf}")


# ---------------------------------------------------------------------------
# Plot sorted inverse flow times + schedule measured in absolute TIME log-log
# ---------------------------------------------------------------------------


# === Plot preparation ===
fig, ax = plt.subplots(figsize=(8, 6))

# Plot each inventory — dots only
for idx, ini_inv in enumerate(invs_to_plot):
    sub = df[df['ini_inventory'] == ini_inv]['flux_time']
    sorted_times = np.sort(sub.values)

    x_values = np.arange(1, len(sorted_times) + 1)
    label = f"IniInv={ini_inv}"

    ax.plot(
        x_values,
        sorted_times,
        label=label,
        color=colors[idx % len(colors)],
        marker='o',
        linestyle='None',  # no connecting line
        markersize=2       # small dot
    )

# === Plot request schedule (sorted) — dashed black line
sorted_schedule_times = np.sort(schedule_df['flux_time'].values)
x_schedule = np.arange(1, len(sorted_schedule_times) + 1)

ax.plot(
    x_schedule,
    sorted_schedule_times,
    label="Request Schedule",
    color='black',
    linestyle='--',
    linewidth=0.8,
    dashes=(6, 4)
)

# === Set log-log axes ===
ax.set_xscale('log')
ax.set_yscale('log')

# === Finalize plot ===
ax.set_xlabel("Index of Sorted Inverse Flow Time (log)", fontsize=12)
ax.set_ylabel("Inverse Flow Time [seconds] (log)", fontsize=12)
ax.set_title("Sorted Inverse Flow Times (log-log)", fontsize=14)
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
ax.legend(frameon=True, edgecolor='black', loc='best')
fig.tight_layout()

# === Save figure ===
plot_path_png = os.path.join(base_dir, "Sorted_Inverse_Flow_Times_Dots_LogLog.png")
plot_path_pdf = os.path.join(base_dir, "Sorted_Inverse_Flow_Times_Dots_LogLog.pdf")

plt.savefig(plot_path_png, dpi=300)
plt.savefig(plot_path_pdf)

plt.close()

print(f"\nSaved log-log plot as PNG: {plot_path_png}")
print(f"Saved log-log plot as PDF: {plot_path_pdf}")




##############################################################################
# Clickspace
##############################################################################

# ---------------------------------------------------------------------------
# Inverse flux of 5 items measured in CLICK SPACE (Collected == 1)
# ---------------------------------------------------------------------------

import os
import csv

def extract_flux_times_click_space(csv_path, agent=1, block=6):
    click_id = 0
    collected_click_ids = []

    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["Agent"]) != agent:
                continue
            if row["Requested"] == '1':
                click_id += 1  # increment click counter
            if row["Collected"] == '1':
                collected_click_ids.append(click_id)

    # Rolling window computation
    flux_clicks = []
    if len(collected_click_ids) >= block:
        for k in range(len(collected_click_ids) - block + 1):
            start = collected_click_ids[k]
            end = collected_click_ids[k + block - 1]
            flux_clicks.append(end - start)

    print(f"Extracted {len(collected_click_ids)} collected events for agent {agent} in click space. Computed {len(flux_clicks)} flux times (block size={block}).")

    return flux_clicks

# === Main ===
if __name__ == "__main__":

    base_dir = "ZI_simulation_outputs"
    agent = 1
    inv_cost = 0.5
    num_agents = 5
    block_size = 6

    # Group 1 — symmetrical
    first_layer = "symmetrical"
    initial_inventories_sym = [1, 2, 3, 4, 5, 10, 20, 100, 1000]

    # Group 2 — first_only
    first_layer_first = "first_only"
    initial_inventories_first = [0.2, 0.4, 0.6, 0.8]

    # Container for ALL flux results in click space
    all_flux_rows = []

    # Process symmetrical
    for ini_inv in initial_inventories_sym:
        folder = os.path.join(base_dir, f"FirstLayer_{first_layer}", f"InventoryCost_{inv_cost}", f"Agents_{num_agents}")
        csv_filename = f"SimLog_Agents_{num_agents}_IniInventory_{ini_inv}_InvCost_{inv_cost}.csv"
        csv_path = os.path.join(folder, csv_filename)

        flux_clicks = extract_flux_times_click_space(csv_path, agent=agent, block=block_size)

        for i, fc in enumerate(flux_clicks, start=1):
            all_flux_rows.append([ini_inv, i, fc])

        print(f"Extracted flux times in click space for symmetrical IniInventory={ini_inv}")

    # Process first_only
    for ini_inv in initial_inventories_first:
        folder = os.path.join(base_dir, f"FirstLayer_{first_layer_first}", f"InventoryCost_{inv_cost}", f"Agents_{num_agents}")
        csv_filename = f"SimLog_Agents_{num_agents}_IniInventory_{ini_inv}_InvCost_{inv_cost}.csv"
        csv_path = os.path.join(folder, csv_filename)

        flux_clicks = extract_flux_times_click_space(csv_path, agent=agent, block=block_size)

        for i, fc in enumerate(flux_clicks, start=1):
            all_flux_rows.append([ini_inv, i, fc])

        print(f"Extracted flux times in click space for first_only IniInventory={ini_inv}")

    # Save the combined CSV in click space
    out_csv_path = os.path.join(base_dir, "Flux_Agent1_ClickSpace.csv")
    with open(out_csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["ini_inventory", "flux_id", "flux_click_span"])
        for row in all_flux_rows:
            w.writerow(row)

    print(f"\nAll flux times in click space saved to {out_csv_path}")





# ---------------------------------------------------------------------------
# Inverse flux of 5 items measured in CLICK SPACE: SCHEDULE (Requested == 1)
# ---------------------------------------------------------------------------

import os
import csv

# === Function to extract request times in click space ===
def extract_request_times_click_space(csv_path, agent=1, block=6):
    click_id = 0
    request_click_ids = []

    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["Agent"]) != agent:
                continue
            if row["Requested"] == '1':
                click_id += 1
                request_click_ids.append(click_id)

    # Rolling window computation
    flux_clicks = []
    if len(request_click_ids) >= block:
        for k in range(len(request_click_ids) - block + 1):
            start = request_click_ids[k]
            end = request_click_ids[k + block - 1]
            flux_clicks.append(end - start)

    print(f"Extracted {len(request_click_ids)} requested events for agent {agent} (click space). Computed {len(flux_clicks)} schedule click spans (block size={block}).")

    return flux_clicks

# === Main ===
if __name__ == "__main__":

    base_dir = "ZI_simulation_outputs"
    agent = 1
    inv_cost = 0.5
    num_agents = 5
    block_size = 6

    first_layer = "symmetrical"
    ini_inv = 1  # ONLY inventory = 1

    # Path to the relevant CSV
    folder = os.path.join(base_dir, f"FirstLayer_{first_layer}", f"InventoryCost_{inv_cost}", f"Agents_{num_agents}")
    csv_filename = f"SimLog_Agents_{num_agents}_IniInventory_{ini_inv}_InvCost_{inv_cost}.csv"
    csv_path = os.path.join(folder, csv_filename)

    # Extract request times in click space
    request_click_spans = extract_request_times_click_space(csv_path, agent=agent, block=block_size)

    # Save schedule click space file
    out_csv_path = os.path.join(base_dir, "Schedule_File_ClickSpace.csv")
    with open(out_csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["flux_id", "flux_click_span"])  # Same logic as Flux_Agent1_ClickSpace.csv
        for i, fc in enumerate(request_click_spans, start=1):
            w.writerow([i, fc])

    print(f"\nSchedule click space file saved to {out_csv_path}")




# ---------------------------------------------------------------------------
# Inverse flux of 5 items: plot of NON-sorted inverse flow times in CLICK SPACE for 0.2, 0.6, 2, 1000 and schedule
# ---------------------------------------------------------------------------

import os
import pandas as pd
import matplotlib.pyplot as plt

# === Matplotlib configuration for scientific style ===
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "axes.edgecolor": "black",
    "axes.linewidth": 1.0,
    "lines.linewidth": 1.5,
    "text.usetex": False
})

# === Configuration ===
base_dir = "ZI_simulation_outputs"
flux_csv_path = os.path.join(base_dir, "Flux_Agent1_ClickSpace.csv")
schedule_csv_path = os.path.join(base_dir, "Schedule_File_ClickSpace.csv")

invs_to_plot = [0.2, 0.6, 2, 1000]
flux_lo = 50
flux_hi = 150

marker_styles = ['^', 's', 'x', 'o']
grey_colors = ['#000000', '#333333', '#666666', '#999999']

# === Load flux CSV ===
df = pd.read_csv(flux_csv_path)

# === Plot preparation ===
fig, ax = plt.subplots(figsize=(7, 5))

# First plot inventory curves
for idx, ini_inv in enumerate(invs_to_plot):
    sub = df[(df['ini_inventory'] == ini_inv) &
             (df['flux_id'] >= flux_lo) &
             (df['flux_id'] <= flux_hi)]

    x_values = sub['flux_id']
    loop_times = sub['flux_click_span']
    label = f"IniInv={ini_inv}"

    ax.plot(
        x_values,
        loop_times,
        label=label,
        color=grey_colors[idx % len(grey_colors)],
        marker=marker_styles[idx % len(marker_styles)],
        markersize=4,
        markeredgewidth=0.8,
        markerfacecolor='white',
        markeredgecolor=grey_colors[idx % len(grey_colors)],
        linewidth=1.5,
        linestyle='-'
    )

# === Add request schedule line ===
schedule_df = pd.read_csv(schedule_csv_path)

# Select same flux_id range
schedule_sub = schedule_df[(schedule_df['flux_id'] >= flux_lo) & (schedule_df['flux_id'] <= flux_hi)]

ax.plot(
    schedule_sub['flux_id'],
    schedule_sub['flux_click_span'],
    label="Request Schedule",
    color='black',
    linestyle='--',
    linewidth=0.8,
    dashes=(6, 4)
)

# === Finalize plot ===
ax.set_xlabel(f"Loop Index ({flux_lo} to {flux_hi})", fontsize=12)
ax.set_ylabel("Inverse Flow Time [click space]", fontsize=12)
ax.set_title(f"Inverse Flow Times in Click Space (Index {flux_lo}–{flux_hi})", fontsize=14)
ax.grid(True)
ax.legend(frameon=True, edgecolor='black')
fig.tight_layout()

# === Save combined plot ===
combined_plot_path = os.path.join(base_dir, "Combined_Flux_Plot_Agent1_ClickSpace.png")
plt.savefig(combined_plot_path, dpi=150)
plt.close()

print(f"Saved combined plot: {combined_plot_path}")



# ---------------------------------------------------------------------------
# MEAN AND STDEV of Inverse Flow Times [click space]
# ---------------------------------------------------------------------------

import os
import pandas as pd

# === Configuration ===
base_dir = "ZI_simulation_outputs"
flux_csv_path = os.path.join(base_dir, "Flux_Agent1_ClickSpace.csv")

# === Inventory groups ===
initial_inventories_sym = [1, 2, 3, 4, 5, 10, 20, 100, 1000]
initial_inventories_first = [0.2, 0.4, 0.6, 0.8]

# === All inventories ===
all_invs = initial_inventories_sym + initial_inventories_first

# === Load flux CSV ===
df = pd.read_csv(flux_csv_path)

# === Compute mean and std per inventory ===
summary_rows = []

for ini_inv in all_invs:
    sub = df[df['ini_inventory'] == ini_inv]
    mean_flux = sub['flux_click_span'].mean()
    std_flux = sub['flux_click_span'].std()

    summary_rows.append({
        "Inventory": ini_inv,
        "Mean Flux Click Span": mean_flux,
        "Std Dev Flux Click Span": std_flux
    })

# === Build summary DataFrame ===
summary_df = pd.DataFrame(summary_rows).sort_values(by='Inventory')

# === Save single summary CSV ===
summary_csv_path = os.path.join(base_dir, "flux_summary_per_inventory_clickspace.csv")
summary_df.to_csv(summary_csv_path, index=False)

# === Print summary to console ===
print("\nFlux Summary Per Inventory (Click Space):\n")
print(summary_df.to_string(index=False))



# ---------------------------------------------------------------------------
# Plot MEAN AND STDEV of Inverse Flow Times [click space] as function of inventory
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Create a 2x2 + 1 wide (5th) panel figure:
# a) Mean Inverse Flow
# b) StdDev Inverse Flow
# c) Mean Inverse Flow (log-log)
# d) StdDev Inverse Flow (log-log, x-limited)
# e) Coefficient of Variation of flux_click_span vs Inventory
# All panels share the same scientific style.
# ---------------------------------------------------------------------------

import os
import pandas as pd
import matplotlib.pyplot as plt

# === Matplotlib configuration ===
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "axes.edgecolor": "black",
    "axes.linewidth": 1.0,
    "lines.linewidth": 1.5,
    "text.usetex": False
})

# === Load data ===
base_dir = "ZI_simulation_outputs"
summary_csv = os.path.join(base_dir, "flux_summary_per_inventory_clickspace.csv")
flux_csv    = os.path.join(base_dir, "Flux_Agent1_ClickSpace.csv")

# Avalanche summary contains:
# 'Inventory', 'Mean Flux Click Span', 'Std Dev Flux Click Span'
sum_df = pd.read_csv(summary_csv).sort_values('Inventory')
# Raw flux for CV
flux_df = pd.read_csv(flux_csv)

# Compute CV of raw flux_click_span
cv_df = (
    flux_df.groupby('ini_inventory')['flux_click_span']
           .agg(Mean='mean', Std='std')
           .reset_index()
)
cv_df['CV'] = cv_df['Std'] / cv_df['Mean']

# Find I_c by peak CV
peak = cv_df.loc[cv_df['CV'].idxmax()]
I_c = peak['ini_inventory']

# Marker & color settings
grey = "#444444"
marker = 'o'

# Create figure with GridSpec
fig = plt.figure(figsize=(10, 12))
gs = fig.add_gridspec(3, 2, height_ratios=[1,1,1.2], hspace=0.5, wspace=0.3)

# Panel a
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(sum_df['Inventory'], sum_df['Mean Flux Click Span'],
         color=grey, marker=marker, markersize=5,
         markerfacecolor='white', markeredgewidth=1.0,
         markeredgecolor=grey)
ax1.set_xlabel("Initial Inventory")
ax1.set_ylabel("Mean Inverse Flow [click space]")
ax1.set_title("a) Mean Inverse Flow\nvs. Initial Inventory")
ax1.grid(True)

# Panel b
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(sum_df['Inventory'], sum_df['Std Dev Flux Click Span'],
         color=grey, marker=marker, markersize=5,
         markerfacecolor='white', markeredgewidth=1.0,
         markeredgecolor=grey)
ax2.set_xlabel("Initial Inventory")
ax2.set_ylabel("StdDev of Inverse Flow [click space]")
ax2.set_title("b) StdDev of Inverse Flow\nvs. Initial Inventory")
ax2.grid(True)

# Panel c
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(sum_df['Inventory'], sum_df['Mean Flux Click Span'],
         color=grey, marker=marker, markersize=5,
         markerfacecolor='white', markeredgewidth=1.0,
         markeredgecolor=grey)
ax3.set_xscale('log')
ax3.set_yscale('log')
ax3.set_xlabel("Initial Inventory (log)")
ax3.set_ylabel("Mean Inverse Flow [click space] (log)")
ax3.set_title("c) Mean Inverse Flow\nvs. Initial Inventory (log-log)")
ax3.grid(True, which='both', linestyle='--', linewidth=0.5)

# Panel d
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(sum_df['Inventory'], sum_df['Std Dev Flux Click Span'],
         color=grey, marker=marker, markersize=5,
         markerfacecolor='white', markeredgewidth=1.0,
         markeredgecolor=grey)
ax4.set_xscale('log')
ax4.set_yscale('log')
ax4.set_xlim(left=sum_df['Inventory'].min(), right=100)
ax4.set_xlabel("Initial Inventory (log)")
ax4.set_ylabel("StdDev of Inverse Flow (log)")
ax4.set_title("d) StdDev of Inverse Flow\nvs. Initial Inventory (log-log)")
ax4.grid(True, which='both', linestyle='--', linewidth=0.5)

# Panel e (span width)
axe = fig.add_subplot(gs[2, :])
axe.plot(cv_df['ini_inventory'], cv_df['CV'],
         color=grey, marker=marker, markersize=5,
         markerfacecolor='white', markeredgewidth=1.0,
         markeredgecolor=grey)
# mark I_c with angled arrow from lower-right
peak_inv = I_c
peak_cv  = peak['CV']
axe.axvline(peak_inv, color='black', linestyle='--', linewidth=1)
axe.annotate(
    rf"$I_c = {peak_inv:.2f}$",
    xy=(peak_inv, peak_cv),                                    # arrow points here
    xytext=(peak_inv * 1.25, peak_cv - 0.08),                  # start text lower
    arrowprops=dict(
        arrowstyle="->",
        color="black",
        shrinkA=0,
        shrinkB=0,
        mutation_scale=16
    ),
    fontsize=12,
    ha='center',
    va='top'   # place the top of the text at xytext, so text extends downward
)
axe.set_xscale('log')
axe.set_xlabel("Initial Inventory")
axe.set_ylabel(r"Coefficient of Variation ($\sigma/\mu$)")
axe.set_title("e) Coefficient of Variation vs. Initial Inventory")
axe.grid(True, which='both', linestyle='--', linewidth=0.5)

# Save and show
graphic = os.path.join(base_dir, "FluxSummary_5Panel_ClickSpace.png")
plt.tight_layout()
plt.savefig(graphic, dpi=300)
plt.show()
print(f"Saved 5-panel figure: {graphic}")






# ---------------------------------------------------------------------------
# Plot 2-panel of sorted inverse flow times + schedule measured in CLICK SPACE
# Panel a) Linear — Panel b) Log-Log
# ---------------------------------------------------------------------------

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === Matplotlib configuration for scientific style ===
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "axes.edgecolor": "black",
    "axes.linewidth": 1.0,
    "lines.linewidth": 1.5,
    "text.usetex": False
})

# === Configuration ===
base_dir = "ZI_simulation_outputs"
flux_csv_path = os.path.join(base_dir, "Flux_Agent1_ClickSpace.csv")
schedule_csv_path = os.path.join(base_dir, "Schedule_File_ClickSpace.csv")

# === Load data ===
df = pd.read_csv(flux_csv_path)
schedule_df = pd.read_csv(schedule_csv_path)

# === Inventory list ===
invs_to_plot = sorted(df['ini_inventory'].unique())

# === Color palette ===
from matplotlib.cm import get_cmap
cmap = get_cmap("tab20")
colors = [cmap(i) for i in range(len(invs_to_plot))]

# === Create 2-panel figure ===
fig, axs = plt.subplots(1, 2, figsize=(14, 6))  # Wider figure

# === Panel a) — Linear ===
for idx, ini_inv in enumerate(invs_to_plot):
    sub = df[df['ini_inventory'] == ini_inv]['flux_click_span']
    sorted_times = np.sort(sub.values)

    x_values = np.arange(1, len(sorted_times) + 1)
    label = f"IniInv={ini_inv}"

    axs[0].plot(
        x_values,
        sorted_times,
        label=label,
        color=colors[idx % len(colors)],
        marker='o',
        linestyle='None',
        markersize=2
    )

# Request schedule — dashed black line
sorted_schedule_times = np.sort(schedule_df['flux_click_span'].values)
x_schedule = np.arange(1, len(sorted_schedule_times) + 1)

axs[0].plot(
    x_schedule,
    sorted_schedule_times,
    label="Request Schedule",
    color='black',
    linestyle='--',
    linewidth=0.8,
    dashes=(6, 4)
)

axs[0].set_xlabel("Index of Sorted Inverse Flow", fontsize=12)
axs[0].set_ylabel("Inverse Flow [click space]", fontsize=12)
axs[0].set_title("a) Sorted Inverse Flows in Click Space", fontsize=14)
axs[0].grid(True)
axs[0].legend(frameon=True, edgecolor='black', loc='best')

# === Panel b) — Log-Log ===
for idx, ini_inv in enumerate(invs_to_plot):
    sub = df[df['ini_inventory'] == ini_inv]['flux_click_span']
    sorted_times = np.sort(sub.values)

    x_values = np.arange(1, len(sorted_times) + 1)
    label = f"IniInv={ini_inv}"

    axs[1].plot(
        x_values,
        sorted_times,
        label=label,
        color=colors[idx % len(colors)],
        marker='o',
        linestyle='None',
        markersize=2
    )

# Request schedule — dashed black line
axs[1].plot(
    x_schedule,
    sorted_schedule_times,
    label="Request Schedule",
    color='black',
    linestyle='--',
    linewidth=0.8,
    dashes=(6, 4)
)

axs[1].set_xscale('log')
axs[1].set_yscale('log')
axs[1].set_xlabel("Index of Sorted Inverse Flow (log)", fontsize=12)
axs[1].set_ylabel("Inverse Flow [click space] (log)", fontsize=12)
axs[1].set_title("b) Sorted Inverse Flows in Click Space (log-log)", fontsize=14)
axs[1].grid(True, which='both', linestyle='--', linewidth=0.5)
axs[1].legend(frameon=True, edgecolor='black', loc='best')

# === Final layout ===
fig.tight_layout()

# === Save figure (PNG only) ===
plot_path_png = os.path.join(base_dir, "Sorted_Inverse_Flow_Times_ClickSpace_2Panel.png")
plt.savefig(plot_path_png, dpi=300)

plt.close()

print(f"\nSaved 2-panel click space plot as PNG: {plot_path_png}")







# ---------------------------------------------------------------------------
# Avalanche analysis in CLICK SPACE — Duration and Size between successive 5s
# ---------------------------------------------------------------------------

import os
import pandas as pd
import numpy as np

# === Configuration ===
base_dir = "ZI_simulation_outputs"
flux_csv_path = os.path.join(base_dir, "Flux_Agent1_ClickSpace.csv")

# === Load data ===
df = pd.read_csv(flux_csv_path)

# === Inventory list ===
invs_to_check = sorted(df['ini_inventory'].unique())



# === Container for avalanche results ===
avalanche_rows = []

# === Process each inventory ===
for ini_inv in invs_to_check:


    print(f"\nProcessing Inventory {ini_inv}...")

    # Subset data for this inventory
    sub = df[df['ini_inventory'] == ini_inv].reset_index(drop=True)
    flux_values = sub['flux_click_span'].values

    # Find indices where flux_click_span == 5
    indices_5 = np.where(flux_values == 5)[0]

    if len(indices_5) < 2:
        print(f"  Not enough 5s found (found {len(indices_5)}). Skipping.")
        continue

    print(f"  Found {len(indices_5)} occurrences of 5 — detecting avalanches...")

    # Loop over successive pairs of 5s
    avalanche_id = 0
    for i in range(len(indices_5) - 1):
        start_idx = indices_5[i]
        end_idx = indices_5[i + 1]
    
        # Skip if the 5s are consecutive — no room for avalanche
        if end_idx - start_idx <= 1:
            continue
    
        # Valid avalanche
        segment = flux_values[start_idx : end_idx + 1]
        duration = end_idx - start_idx - 1
    
        # Subtract baseline (5)
        segment_adj = segment - 5
    
        # Compute size on adjusted segment
        size = np.trapz(segment_adj)
    
        avalanche_rows.append({
            "Inventory": ini_inv,
            "Avalanche_ID": avalanche_id + 1,
            "Duration": duration,
            "Size": size
        })
    
        avalanche_id += 1



# === Build DataFrame ===
avalanche_df = pd.DataFrame(avalanche_rows)

# === Save to CSV ===
avalanche_csv_path = os.path.join(base_dir, "flux_avalanches_clickspace.csv")
avalanche_df.to_csv(avalanche_csv_path, index=False)

# === Print summary ===
print(f"\nAll avalanche data saved to: {avalanche_csv_path}")

print("\nAvalanche DataFrame (first 10 rows):")
print(avalanche_df.head(10).to_string(index=False))


# CHECK If CALCULATIONS ARE CORRECT

import os
import pandas as pd

# === File paths ===
base_dir = "ZI_simulation_outputs"
flux_csv = os.path.join(base_dir, "Flux_Agent1_ClickSpace.csv")
avalanche_csv = os.path.join(base_dir, "flux_avalanches_clickspace.csv")

# === Load and filter first 50 flux rows for inventory 1 ===
flux_rows = pd.read_csv(flux_csv)
flux_subset = flux_rows[flux_rows['ini_inventory'] == 1].head(80)

print("\nFirst 50 rows of flux data for inventory == 1:")
print(flux_subset.to_string(index=False))

# === Load and filter first 10 avalanche rows for inventory 1 ===
avalanche_rows = pd.read_csv(avalanche_csv)
avalanche_subset = avalanche_rows[avalanche_rows['Inventory'] == 1].head(10)

print("\nFirst 10 rows of avalanche data for inventory == 1:")
print(avalanche_subset.to_string(index=False))




# -----------------------------------------------------------
# FINAL 4-PANEL PLOT: Mean Avalanche Statistics vs Initial Inventory
# -----------------------------------------------------------
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === Matplotlib configuration for scientific style ===
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.edgecolor": "black",
    "axes.linewidth": 1.0,
    "lines.linewidth": 1.5,
    "text.usetex": False
})

# === Load avalanche CSV ===
base_dir = "ZI_simulation_outputs"
avalanche_csv_path = os.path.join(base_dir, "flux_avalanches_clickspace.csv")
avalanche_df = pd.read_csv(avalanche_csv_path)

# === Group by Inventory → compute mean Duration and mean Size ===
group_stats = avalanche_df.groupby("Inventory").agg({
    "Duration": ["mean"],
    "Size": ["mean"]
}).reset_index()

# Flatten columns
group_stats.columns = ["Inventory", "Mean Duration", "Mean Size"]

# === Prepare variables for plotting ===
inventory_values = group_stats["Inventory"].values
mean_duration_values = group_stats["Mean Duration"].values
mean_size_values = group_stats["Mean Size"].values

# === Grey color for consistency ===
grey_color = "#444444"
marker_style = 'o'

# === 4-panel plot ===
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Panel a) Mean Duration - linear
axes[0, 0].plot(inventory_values, mean_duration_values,
                color=grey_color,
                marker=marker_style,
                markersize=5,
                markerfacecolor='white',
                markeredgewidth=1.0,
                markeredgecolor=grey_color,
                linestyle='-')
axes[0, 0].set_xlabel("Initial Inventory")
axes[0, 0].set_ylabel("Mean Avalanche Duration")
axes[0, 0].grid(True)
axes[0, 0].set_title("a) Mean Avalanche Duration\nvs. Initial Inventory")

# Panel b) Mean Size - linear
axes[0, 1].plot(inventory_values, mean_size_values,
                color=grey_color,
                marker=marker_style,
                markersize=5,
                markerfacecolor='white',
                markeredgewidth=1.0,
                markeredgecolor=grey_color,
                linestyle='-')
axes[0, 1].set_xlabel("Initial Inventory")
axes[0, 1].set_ylabel("Mean Avalanche Size")
axes[0, 1].grid(True)
axes[0, 1].set_title("b) Mean Avalanche Size\nvs. Initial Inventory")

# Panel c) Mean Duration - log-log
axes[1, 0].plot(inventory_values, mean_duration_values,
                color=grey_color,
                marker=marker_style,
                markersize=5,
                markerfacecolor='white',
                markeredgewidth=1.0,
                markeredgecolor=grey_color,
                linestyle='-')
axes[1, 0].set_xscale("log")
axes[1, 0].set_yscale("log")
axes[1, 0].set_xlabel("Initial Inventory (log)")
axes[1, 0].set_ylabel("Mean Avalanche Duration (log)")
axes[1, 0].grid(True, which='both', linestyle='--', linewidth=0.5)
axes[1, 0].set_title("c) Mean Avalanche Duration\nvs. Initial Inventory (log-log)")

# Panel d) Mean Size - log-log
axes[1, 1].plot(inventory_values, mean_size_values,
                color=grey_color,
                marker=marker_style,
                markersize=5,
                markerfacecolor='white',
                markeredgewidth=1.0,
                markeredgecolor=grey_color,
                linestyle='-')
axes[1, 1].set_xscale("log")
axes[1, 1].set_yscale("log")
axes[1, 1].set_xlabel("Initial Inventory (log)")
axes[1, 1].set_ylabel("Mean Avalanche Size (log)")
axes[1, 1].grid(True, which='both', linestyle='--', linewidth=0.5)
axes[1, 1].set_title("d) Mean Avalanche Size\nvs. Initial Inventory (log-log)")

# Optional: if you want, you can limit x-axis for panel d) like your previous style
# axes[1, 1].set_xlim(left=inventory_values.min(), right=100)

# === Final layout ===
plt.tight_layout()

# === Save figure ===
png_path = os.path.join(base_dir, "FluxAvalancheSummary_Panels_ClickSpace.png")
plt.savefig(png_path, dpi=300)
print(f"\nSaved high-quality PNG: {png_path}")

# Close the figure
plt.close()


# ---------------------------------------------------------------------------
# FULL 6-PANEL CRITICALITY FIGURE (a–f)
# ---------------------------------------------------------------------------

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === Matplotlib configuration ===
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "axes.edgecolor": "black",
    "axes.linewidth": 1.0,
    "lines.linewidth": 1.5,
    "text.usetex": False
})

# === Paths ===
base_dir       = "ZI_simulation_outputs"
avalanche_csv  = os.path.join(base_dir, "flux_avalanches_clickspace.csv")
flux_csv       = os.path.join(base_dir, "Flux_Agent1_ClickSpace.csv")
schedule_csv   = os.path.join(base_dir, "Schedule_File_ClickSpace.csv")
summary_csv    = os.path.join(base_dir, "flux_summary_per_inventory_clickspace.csv")

# === Load data ===
avalanche_df = pd.read_csv(avalanche_csv)
flux_df      = pd.read_csv(flux_csv)
schedule_df  = pd.read_csv(schedule_csv)
sum_df       = pd.read_csv(summary_csv).sort_values("Inventory").reset_index(drop=True)

# === Panel a/c/d data ===
group_stats  = avalanche_df.groupby("Inventory").agg(
    Num_Avalanches=("Avalanche_ID","count"),
    Mean_Duration =("Duration",   "mean"),
    Mean_Size     =("Size",       "mean")
).reset_index()
B_all        = group_stats["Inventory"].values
num_avals    = group_stats["Num_Avalanches"].values
dur_means    = group_stats["Mean_Duration"].values
size_means   = group_stats["Mean_Size"].values

# === Identify I_c by peak avalanche count ===
I_c = B_all[np.argmax(num_avals)]

# === Panel e/f sensitivity data ===
# use same B_all, dur_means, size_means
dtaudB = np.abs(np.gradient(dur_means, B_all))
dSdB   = np.abs(np.gradient(size_means, B_all))

# === Panel b parameters ===
flux_lo, flux_hi = 50, 150
invs_to_plot     = [0.2, I_c, 2, 1000]
colors_b         = ['#9b59b6','#f39c12','#1abc9c','#ff66cc']
markers_b        = ['^','s','x','o']

# === Styling ===
grey = "#444444"
mk   = dict(marker="o", markersize=5,
            markerfacecolor="white",
            markeredgewidth=1, markeredgecolor=grey)

# === Create 3×2 figure ===
fig, axs = plt.subplots(3, 2, figsize=(11, 12))
plt.subplots_adjust(hspace=0.5, wspace=0.3)

# ----- Panel a) Number of Avalanches vs Inventory -----
ax = axs[0,0]
ax.plot(B_all, num_avals, color=grey, linestyle="-", **mk)
ax.set_xscale("log")
ax.set_xlabel("Initial Inventory (log)")
ax.set_ylabel("Number of Avalanches")
ax.set_title("a) Number of Avalanches\nvs. Initial Inventory")
ax.grid(True, which="both", linestyle="--", linewidth=0.5)
ax.axvline(I_c, color="black", linestyle="--", linewidth=1)
ax.annotate(rf"$I_c = {I_c:.2f}$",
            xy=(I_c, num_avals.max()),
            xytext=(I_c*2, num_avals.max()*0.9),
            arrowprops=dict(arrowstyle="->", color="black", shrinkA=0, shrinkB=0),
            ha="left", va="top")

# ----- Panel b) Inverse Flow Times in Click Space -----
ax = axs[0,1]
labels = [
          rf"$I_{{<c}} = {invs_to_plot[0]:.1f}$",
          rf"$I_c = {invs_to_plot[1]:.2f}$",
          rf"$I_{{>c}} = {invs_to_plot[2]:.1f}$",
          rf"$I_{{>>c}} = {invs_to_plot[3]:.0f}$"]
for i, inv in enumerate(invs_to_plot):
    dfsub = flux_df[(flux_df.ini_inventory==inv)&
                    (flux_df.flux_id.between(flux_lo,flux_hi))]
    ax.plot(dfsub.flux_id, dfsub.flux_click_span,
            label=labels[i],
            color=colors_b[i],
            marker=markers_b[i],
            markersize=4,
            markerfacecolor='white',
            markeredgewidth=0.8,
            markeredgecolor=colors_b[i],
            linewidth=1.5, linestyle='-')
sched = schedule_df[schedule_df.flux_id.between(flux_lo,flux_hi)]
ax.plot(sched.flux_id, sched.flux_click_span,
        label="Click Schedule", color='black',
        linestyle='--', linewidth=0.8, dashes=(6,4))
ax.set_xlabel(f"Loop Index ({flux_lo}–{flux_hi})")
ax.set_ylabel("Inverse Flow Time [click space]")
ax.set_title("b) Inverse Flow Times\nin Click Space")
ax.grid(True, linestyle="--", linewidth=0.5)
ax.legend(loc="upper right", frameon=True, edgecolor='black', fontsize=9)

# ----- Panel c) Mean Avalanche Duration vs Inventory (log-log) -----
ax = axs[1,0]
mask = B_all<=20
ax.plot(B_all[mask], dur_means[mask], color=grey, linestyle="-", **mk)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Initial Inventory (log)")
ax.set_ylabel("Mean Avalanche Duration (log)")
ax.set_title("c) Mean Avalanche Duration\nvs. Initial Inventory (log-log)")
ax.grid(True, which="both", linestyle="--", linewidth=0.5)
ax.axvline(I_c, color="black", linestyle="--", linewidth=1)
ax.annotate(rf"$I_c = {I_c:.2f}$",
            xy=(I_c, dur_means[list(B_all).index(I_c)]),
            xytext=(I_c*1.5, dur_means[list(B_all).index(I_c)]*1.5),
            arrowprops=dict(arrowstyle="->", color="black", shrinkA=0, shrinkB=0),
            fontsize=12, ha="left", va="bottom")

# ----- Panel d) Mean Avalanche Size vs Inventory (log-log) -----
ax = axs[1,1]
ax.plot(B_all[mask], size_means[mask], color=grey, linestyle="-", **mk)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Initial Inventory (log)")
ax.set_ylabel("Mean Avalanche Size (log)")
ax.set_title("d) Mean Avalanche Size\nvs. Initial Inventory (log-log)")
ax.grid(True, which="both", linestyle="--", linewidth=0.5)
ax.axvline(I_c, color="black", linestyle="--", linewidth=1)
ax.annotate(rf"$I_c = {I_c:.2f}$",
            xy=(I_c, size_means[list(B_all).index(I_c)]),
            xytext=(I_c*1.5, size_means[list(B_all).index(I_c)]*1.5),
            arrowprops=dict(arrowstyle="->", color="black", shrinkA=0, shrinkB=0),
            fontsize=12, ha="left", va="bottom")

# ----- Panel e) Sensitivity: d<D>/dB -----
ax = axs[2,0]
ax.plot(B_all, dtaudB, color=grey, linestyle="-", **mk)
ax.set_xscale("log")
ax.set_xlabel("Initial Inventory")
ax.set_ylabel(r"$|d\langle\tau\rangle/dB|$")
ax.set_title("e) Sensitivity of Mean Avalanche Duration")
ax.grid(True, which="both", linestyle="--", linewidth=0.5)
ax.axvline(I_c, color="black", linestyle="--", linewidth=1)
idx = list(B_all).index(I_c)
ax.annotate(rf"$I_c = {I_c:.2f}$",
            xy=(I_c, dtaudB[idx]),
            xytext=(I_c*1.4, dtaudB[idx] + 200),
            arrowprops=dict(arrowstyle="->", color="black", shrinkA=0, shrinkB=0),
            fontsize=12, ha="center", va="bottom")

# ----- Panel f) Sensitivity: d<S>/dB -----
ax = axs[2,1]
ax.plot(B_all, dSdB, color=grey, linestyle="-", **mk)
ax.set_xscale("log")
ax.set_xlabel("Initial Inventory")
ax.set_ylabel(r"$|d\langle S\rangle/dB|$")
ax.set_title("f) Sensitivity of Mean Avalanche Size")
ax.grid(True, which="both", linestyle="--", linewidth=0.5)
ax.axvline(I_c, color="black", linestyle="--", linewidth=1)
ax.annotate(rf"$I_c = {I_c:.2f}$",
            xy=(I_c, dSdB[idx]),
            xytext=(I_c*1.4, dSdB[idx] + 1000),
            arrowprops=dict(arrowstyle="->", color="black", shrinkA=0, shrinkB=0),
            fontsize=12, ha="center", va="bottom")

# ----- Finalize & save -----
plt.tight_layout()
outpath = os.path.join(base_dir, "FluxAvalanche_6Panel_Criticality.png")
plt.savefig(outpath, dpi=300)
plt.close()

print(f"Saved 6-panel figure: {outpath}")





