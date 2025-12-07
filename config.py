import os

# System Constants
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(PROJECT_ROOT, 'corruption.db')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

# Agent Counts
NUM_CORRUPT_COPS = 5
NUM_HONEST_COPS = 2
TOTAL_AGENTS = 1 + 1 + NUM_CORRUPT_COPS + NUM_HONEST_COPS  # Chief + IA + Corrupt + Honest

# Simulation Parameters
# TRAINING: Increase to 3000+ for visible Q-Learning convergence
NUM_EPISODES = 10000
INSPECTION_FREQUENCY = 50

# Risk Factors
WITNESS_RISK_FACTOR = 0.15
IA_RISK_MULTIPLIER = 0.40
LOCATION_RISK_WEIGHT = 0.20

# Q-LEARNING PARAMETERS (DQN UPGRADE)
ALPHA = 0.001       # Learning Rate (LR)
GAMMA = 0.99        # Discount Factor
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.9995
BATCH_SIZE = 64
MEMORY_SIZE = 10000
TARGET_UPDATE_FREQ = 100
HIDDEN_DIM = 64

BRAIN_DIR = os.path.join(PROJECT_ROOT, 'brains')

# Actions
WITNESS_BRIBE_COST = 500  # Cost to silence a witness before accepting

# Rewards for Q-Learning

# Rewards for Q-Learning
REWARD_SUCCESS_FACTOR = 0.005 # Offer * 0.005
REWARD_CAUGHT = -50
REWARD_REJECTED = -1
REWARD_KILLED = -200          # Massive penalty (Agent died)

# Violence Settings
VIOLENCE_SCALAR = 0.03 # Probability of getting killed = Severity * 0.03 (Max ~30% for Murder)

# CRIME DEFINITIONS (New Realism Update)
# Format: 'Name': (Severity 1-10, Min_Offer, Max_Offer)
CRIME_TYPES = {
    'Traffic_Violation': (1, 500, 2000),      # Low Risk, Low Reward
    'Petty_Theft':       (3, 3000, 8000),     # Medium
    'Assault':           (6, 10000, 25000),   # High
    'Drug_Trafficking':  (8, 30000, 60000),   # Very High
    'Murder':            (10, 100000, 500000) # Extreme Risk
}
SEVERITY_RISK_MULTIPLIER = 0.08 # Risk increases by 8% per severity point

# Stats for Hierarchy
CORRUPTION_GAIN_SUCCESS = 2
CORRUPTION_LOSS_CAUGHT = -20
PARANOIA_GAIN_CAUGHT = 0.3
PARANOIA_LOSS_SUCCESS = -0.05
LOYALTY_LOSS_CAUGHT = -15

# Global Alert Settings (Witness Backup)
ALERT_DECAY_RATE = 0.05       # Cooling down per episode
ALERT_RISK_ADDITION = 0.25    # +25% Detection Chance at MAX HEAT

# Thresholds
SUSPICION_THRESHOLD = 70.0
EVIDENCE_THRESHOLD = 60.0

# Initial Values
INITIAL_INTEGRITY_HONEST = 90
INITIAL_CORRUPTION_MIN = 30
INITIAL_CORRUPTION_MAX = 60

# Visualization Settings
DPI = 250
FIG_SIZE_WIDE = (10, 6)
FIG_SIZE_SQUARE = (8, 8)
