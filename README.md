# 🚔 Multi-Agent Police Corruption Simulation

> **"Where death is just a learning experience, and corruption is a survival strategy."**

![Status](https://img.shields.io/badge/Status-Live-green)
![Python](https://img.shields.io/badge/Python-3.9+-blue)
![AI](https://img.shields.io/badge/AI-Deep%20Reinforcement%20Learning%20(DQN)-red)
![Framework](https://img.shields.io/badge/Framework-PyTorch-orange)

## 📜 Overview
**CorruptionMARL** is a sophisticated **Multi-Agent Reinforcement Learning (MARL)** simulation that models the emergence of systemic corruption within a hierarchical police force. 

Unlike standard RL environments where agents reset after every game, this system implements **Inter-generational Knowledge Transfer**. When an officer is killed by a cartel or criminals, their learned policy (stored in a `.pth` Neural Network file) is **physically inherited** by a new recruit. This creates an "immortal" institutional memory where the police force evolves from "Naive & Honest" to "Smart & Corrupt" over thousands of episodes.

---

## 🚀 Key Features (The "Wow" Factor)

### 🧠 1. Deep Q-Network (DQN) Brains
*   **Neural Beings:** Agents are not simple scripts. They utilize **PyTorch Deep Neural Networks** to process complex 5-dimensional state vectors (Witnesses, IA Presence, Bribe Amount, Crime Severity, Global Heat).
*   **Experience Replay:** Agents remember past successes and failures, learning from batches of memories to stabilize decision-making.
*   **Legacy System:** Death is not the end. The neural weights of a fallen officer are loaded into their successor, ensuring that *survival strategies* persist across generations.

### 🔄 2. Continuous "Resume" Training
*   **Persistent World:** The simulation saves its state in `training_state.json` and the SQLite database.
*   **Resume Anytime:** You can stop the simulation at Episode 5,000 and resume exactly where you left off. The system loads all active agents and their specific stats effectively.
*   **Infinite Learning:** Train for 10,000, 20,000, or 100,000 episodes to see advanced behaviors emerge.

### 🚨 3. Witness Backup & Global Heat ("Bandit Down")
*   **Witness Reaction:** If an officer is killed in front of witnesses, they call for backup.
*   **Global High Alert:** This triggers a **System-Wide Heat Wave**. For the next several episodes, the `alert_level` spikes to 100%, increasing the risk of getting caught for *every* officer on the force.
*   **Emergent Behavior:** Smart agents learn to universally **REJECT** bribes during these heat waves to survive.

### ⚖️ 4. "Grey" Morality & Hierarchy
*   **No "Good" Guys:** The system doesn't enforce binary morality.
    *   **Police Chief:** Can't be fired, but monitors subordinates. If a cop earns too much black money, the Chief **Blackmails** them for a cut.
    *   **Internal Affairs (IA):** Tasked with catching corrupt cops, but will **Cover Up** investigations if the bribe offer is high enough.

---

## 🛠️ System Architecture

The simulation runs on a custom **Event-Driven Architecture**:

| Component | Role | Description |
| :--- | :--- | :--- |
| **`main.py`** | The Engine | Orchestrates the simulation, handles "Resume" logic, manages the Global Heat cycle, and Agent Respawn. |
| **`agents/corrupt_cop.py`** | The Agent | Implements the DQN Agent. Decides actions (`ACCEPT`, `REJECT`, `CLEAN_AND_ACCEPT`) and learns from rewards. |
| **`agents/dqn_model.py`** | The Brain | A PyTorch `nn.Module` (Input 5 -> Hidden 64 -> Output 3) representing the officer's policy. |
| **`environment/game_world.py`** | The World | Procedurally generates diverse scenarios (Traffic Stops vs Cartel Hits) with varying risks and rewards. |
| **`training_state.json`** | The Save File | Persists the global episode count to allow seamless continuous training. |
| **`database/simulation.db`** | The Memory | A robust SQLite database logging every bribe, death, order, and investigation for analysis. |

---

## 📊 Installation & Usage

### Prerequisites
*   Python 3.8+
*   `torch` (PyTorch)
*   `numpy`
*   `matplotlib`

### 1. Installation
```bash
git clone https://github.com/lavyadamania/Multi-Agent-Police-Corruption-Simulation.git
cd Multi-Agent-Police-Corruption-Simulation
pip install torch numpy matplotlib
```

### 2. Run the Simulation
```bash
python CorruptionMARL_Complete/main.py
```
*   **First Run:** Starts from Episode 0 (Fresh Database).
*   **Subsequent Runs:** Automatically detects `training_state.json` and RESUMES from the last episode.

### 3. Resetting (Fresh Start)
To delete all progress and start from zero:
*   **Windows (PowerShell):**
    ```powershell
    Remove-Item "CorruptionMARL_Complete/brains/*.pth" -ErrorAction SilentlyContinue
    Remove-Item "CorruptionMARL_Complete/*.db" -ErrorAction SilentlyContinue
    Remove-Item "CorruptionMARL_Complete/*.json" -ErrorAction SilentlyContinue
    ```

---

## 📈 Results & Visualization

Events are logged to `simulation.db`. The system auto-generates graphs for:
1.  **Corruption Trend:** Tracking average corruption scores over time.
2.  **Risk vs Reward:** Correlation between bribe amounts and acceptance rates.
3.  **Survival of the Fittest:** Comparing the lifespan of "Greedy" vs "Cautious" agents.

---

## 🔬 Scientific Potential

This project serves as a testbed for research into:
1.  **Evolutionary Stability Strategies (ESS):** How does "Legacy Inheritance" speed up convergence compared to standard Exploration?
2.  **Systemic Resilience:** How quickly does the network recover after a "Purge" (High Heat Event)?
3.  **Incentive Mechanism Design:** Testing if higher salaries actually reduce corruption when "Greed" is an intrinsic agent parameter.

---

> **Author:** Lavya Damania
> 
> *Built with Python, PyTorch, and a bit of Machiavellian logic.* 🐍
