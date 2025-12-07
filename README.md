# 🚔 CorruptionMARL: Evolutionary Multi-Agent Police System

> **"Where death is just a learning experience, and corruption is a survival strategy."**

![Status](https://img.shields.io/badge/Status-Complete-green)
![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Type](https://img.shields.io/badge/AI-Deep%20Reinforcement%20Learning%20(DQN)-red)

## 📜 Overview
**CorruptionMARL** is a sophisticated **Multi-Agent Reinforcement Learning (MARL)** simulation that models the emergence of systemic corruption within a hierarchical police force. Unlike standard RL environments where agents reset, this system implements **Inter-generational Knowledge Transfer**—when an officer is killed in the line of duty, their learned policy (**Deep Neural Network**) is inherited by a new recruit, creating an "immortal" institutional memory.

---

## 🚀 Key Features ( The "Wow" Factor)

### 🧠 1. Deep Q-Network (DQN) Brains [New]
*   **Neural Architecture:** Officers use **PyTorch-based Deep Neural Networks** to process complex state vectors (5-dimensional input).
*   **Experience Replay:** Agents remember past interactions and learn from batches, stabilizing the learning process.
*   **Legacy Respawn:** Upon death, the `.pth` model weights are transferred to the successor.

### 🚨 2. Witness Backup & Global Heat ("Bandit Down")
*   **Systemic Reaction:** If an officer is killed in front of witnesses, a **"Backup Call"** is triggered.
*   **Global High Alert:** The entire simulation enters a `High Heat` state where detection risk increases by **25%** for ALL agents.
*   **Emergent Behavior:** Agents learn to "lay low" (reject bribes) during heat waves to survive.

### 💸 3. Dynamic Dark Economy
*   **Context-Aware Bribes:** Bribe offers are not random; they are calculated based on **Asset Value** (stolen goods), **Criminal Wealth**, and **Crime Severity**.
*   **Hierarchical Blackmail:** The **Police Chief** (Agent #0) monitors subordinates. If an officer earns too much illicit money, the Chief extorts a cut instead of firing them.

---

## 🛠️ System Architecture

The simulation runs on a custom **Event-Driven Architecture**:

| Component | Role | Logic |
| :--- | :--- | :--- |
| **`main.py`** | The Engine | Runs the 3000-episode loop, manages life-cycles (Kill/Spawn), and triggers Global Heat. |
| **`agents/corrupt_cop.py`** | The Agent | Uses **Deep Q-Network (DQN)** with Replay Memory. State: `[Witness, IA, Offer, Severity, Alert]`. |
| **`agents/dqn_model.py`** | The Brain | PyTorch Neural Network (Input 5 -> Hidden 64 -> Output 3). |
| **`environment/game_world.py`** | The World | Procedurally generates crime scenarios ranging from *Traffic Violations* to *Cartel Murders*. |
| **`database/schema.sql`** | The Memory | A robust **SQLite** backend that logs every single transaction, death, and interaction for analysis. |

---

## 📊 installation & Usage

### Prerequisites
*   Python 3.8+
*   `numpy`, `matplotlib`, `sqlite3`

### Run the Simulation
```bash
# Clone the repository
git clone https://github.com/yourusername/CorruptionMARL.git

# Run the main simulation
python CorruptionMARL_Complete/main.py
```

### View Results
After the simulation completes (approx 3000 episodes):
1.  Check the console for **"KILLED"** and **"RECRUITMENT"** logs.
2.  Navigate to `CorruptionMARL_Complete/results/` to see generated graphs:
    *   `corruption_trend.png`: Visualizing the rise/fall of systemic corruption.
    *   `earnings_distribution.png`: Wealth inequality among officers.

---

## 🔬 Research Potential (For Admissions/Thesis)

This project addresses several open problems in Multi-Agent Systems:

1.  **Evolutionary Stability:** How does "inheritance" affect the convergence speed of cooperative (or corrupt) policies?
2.  **Systemic Resilience:** Analyzing how the system recovers from "shocks" (e.g., removal of key agents or sudden high-risk phases).
3.  **Mechanism Design:** Designing incentive structures (salaries vs. bribes) to curb corruption in autonomous agent networks.

---

> Built with ❤️ and 🐍 Python.
