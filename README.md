
# Zero-Intelligence Supply Chain Simulation – Fragility Calibration

This repository contains the full simulation and analysis code for the Zero-Intelligence (ZI) model used in the MSc thesis _"Timeliness Criticality in Supply Chains"_ by Merel de Leur (2025). The simulation framework was developed to isolate structural sources of systemic fragility in decentralized supply networks.

---

## Features

- Stylized circular supply chain simulation with zero-intelligence agents
- Systematic variation of inventory, initial inventory distribution, number of agents in the ring, and holding costs
- Temporal event logging and balance tracking for each agent
- Extraction of inverse flow times, delays, and avalanches
- Publication-quality plots for fragility diagnostics

---

## Structure

```
ZI_simulation_outputs/           # All generated data and plots
├── FirstLayer_<type>/          # Inventory distribution style (e.g. 'first_only')
│   └── InventoryCost_<val>/
│       └── Agents_<N>/         # Simulation results by agent count
│           ├── SimLog_*.csv   # Full timestamped event logs
│           └── *.pdf          # Diagnostic plots (only for 5 agents)
├── Flux_Agent1.csv             # Inverse flow times (time-based)
├── Flux_Agent1_ClickSpace.csv  # Inverse flow times (click-based)
├── Schedule_File_Time.csv      # Baseline request schedule in seconds
├── Schedule_File_ClickSpace.csv# Baseline request schedule in clicks
├── flux_summary_per_inventory.csv
├── flux_summary_per_inventory_clickspace.csv
├── flux_avalanches_clickspace.csv
├── FluxSummary_Panels.png
├── FluxSummary_5Panel_ClickSpace.png
└── FluxAvalanche_6Panel_Criticality.png
```

---

## Requirements

Install dependencies using:

```bash
pip install numpy pandas matplotlib
```

> Tested with Python 3.10+

---

## Running the Code

To run the full simulation and analysis pipeline:

```bash
python run_simulations.py
```

The script executes all parameter combinations and saves outputs into structured folders.  
You can modify parameters like:

```python
first_layer_distributions = ["first_only"]
inventory_costs = [0,05, 0.5, 1.0, 1,5]
agent_counts = [4, 5, 6, 10, 20, 40]
initial_inventories = [1, 2, 3, 4, 5, 10, 20]
```

---

## Key Outputs

- **Event Logs**: Inventory, balance, and transactions at each time tick
- **Flux measurements**: Time taken to collect 5 items (agent-level)
- **Schedule benchmarks**: Baseline request rhythm
- **Delay and avalanche metrics**: Duration and size of disruptions
- **Criticality plots**: Mean flows, CV peaks, and avalanche sensitivities

---

## Model Description

The model represents a closed-loop supply chain of `N` agents. Each agent:

- Has a single upstream supplier and downstream customer
- Holds inventory and a monetary balance (initially 30 ECU)
- Requests items with asynchronous intervals (uniformly random: 0.01–4.00s)
- Incurs holding costs every second (`c × inventory`)

### Request Rules:
- Agent requests from its upstream supplier
- If supplier has stock → transfer occurs:  
  - Supplier loses 1 item and gains 10 ECU  
  - Requesting agent gains 1 item
- If supplier is empty → no transfer
- Requests cost 0 ECU but missed deliveries impose opportunity cost

### Inventory Initialization Modes:
- **Symmetrical**: Equal inventory for all agents
- **First Only**: All inventory held by agent 1
- **Random**: Total inventory distributed multinomially

### Core Parameters:
- `initial_inventory` — total stock per agent (or total)
- `price_inventory` — per-unit holding cost per second
- `N` — number of agents in the circular chain
- `first_layer` — distribution strategy at time zero

The simulation is non-adaptive: agents do not learn or strategize. Any instability is structural or temporal.

---

## Simulation Output

Each simulation produces a full **CSV log** of the system evolution.  
Columns include:

- `Time Step`, `Agent`, `Requested`, `Collected`, `Completed`
- `Inventory`, `Balance`, `Costs`, `Earnings`
- `Bankrupt` flags

If `num_agents == 5`, agent-specific plots (`.pdf`) are also created.

---

## Analysis Modules

### 1. Inverse Flow Time
- Measures time to collect 5 items (agent 1)
- Rolling block size of 5 items
- Saved to `Flux_Agent1.csv` (time) and `Flux_Agent1_ClickSpace.csv` (clicks)

### 2. Schedule Benchmark
- Extracts expected click rhythm baseline
- Used for visual reference in plots

### 3. Statistical Summary
- Mean and standard deviation per inventory level
- Plotted both linearly and log-log:
  - `FluxSummary_Panels.png`
  - `FluxSummary_5Panel_ClickSpace.png`

### 4. Avalanche Detection
- Events between two baseline (5) flows
- Compute `duration` (count) and `size` (area under curve)
- Stored in `flux_avalanches_clickspace.csv`

### 5. Criticality Diagnostics
- Peak CV identifies critical inventory \( I_c \)
- Six-panel visualization:
  - Avalanche frequency
  - Mean size & duration
  - Sensitivities (`dS/dB`, `dτ/dB`)
  - `FluxAvalanche_6Panel_Criticality.png`

---

## Thesis Figures

All generated plots are thesis-ready:

- Inverse flow comparisons (raw + sorted)
- Avalanche statistics vs inventory
- Coefficient of variation peaks
- 6-panel criticality overview figure

---

## Citation

If you use this code (or parts of it), please cite:

> de Leur, M. J. M. (2025). _Timeliness Criticality in Supply Chains_. MSc Finance Thesis. Vrije Universiteit Amsterdam.

---

## Contact

Merel de Leur  
mjm.deleur@gmail.com
