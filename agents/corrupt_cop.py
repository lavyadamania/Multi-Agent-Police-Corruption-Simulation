import random
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
from agents.dqn_model import DQN
from config import (
    WITNESS_RISK_FACTOR, IA_RISK_MULTIPLIER, LOCATION_RISK_WEIGHT,
    CORRUPTION_GAIN_SUCCESS, CORRUPTION_LOSS_CAUGHT,
    PARANOIA_GAIN_CAUGHT, PARANOIA_LOSS_SUCCESS, LOYALTY_LOSS_CAUGHT,
    ALPHA, GAMMA, EPSILON_START, EPSILON_MIN, EPSILON_DECAY,
    BATCH_SIZE, MEMORY_SIZE, TARGET_UPDATE_FREQ, HIDDEN_DIM,
    BRAIN_DIR
)

# Define Transition tuple for Replay Memory
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class CorruptCop:
    def __init__(self, agent_id, name, personality, corruption_score):
        self.agent_id = agent_id
        self.name = name
        self.cop_type = 'corrupt'
        self.rank = 'constable'
        self.personality = personality
        self.corruption_score = corruption_score
        self.paranoia_level = 0.0
        self.loyalty_score = 100.0
        
        # Stats
        self.times_bribed = 0
        self.times_caught = 0
        self.total_money_earned = 0.0
        
        # DQN Setup
        self.input_dim = 5  # Witnesses, IA, Offer, Severity, Alert
        self.output_dim = 3 # ACCEPT, REJECT, CLEAN_AND_ACCEPT
        
        # Device Check
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy_net = DQN(self.input_dim, self.output_dim, HIDDEN_DIM).to(self.device)
        self.target_net = DQN(self.input_dim, self.output_dim, HIDDEN_DIM).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=ALPHA)
        self.memory = deque(maxlen=MEMORY_SIZE)
        
        self.epsilon = EPSILON_START
        self.steps_done = 0
        
        self.last_state = None
        self.last_action = None

        # Try to load existing brain
        self.load_brain()

    def get_state_vector(self, state):
        """
        Converts state dictionary to Normalized Tensor.
        [Witnesses(norm), IA(0/1), Offer(norm), Severity(norm), Alert(norm)]
        """
        w_norm = min(state['witnesses'] / 5.0, 1.0)
        ia_val = 1.0 if state['ia_nearby'] else 0.0
        offer_norm = min(state['offer'] / 500000.0, 1.0) # Max 5 Lakh
        sev_norm = state['severity'] / 10.0
        alert_val = state.get('alert_level', 0.0)
        
        return torch.tensor([w_norm, ia_val, offer_norm, sev_norm, alert_val], dtype=torch.float32).to(self.device)

    def decide_bribe(self, state):
        state_tensor = self.get_state_vector(state)
        self.last_state = state_tensor
        
        actions = ['ACCEPT', 'REJECT', 'CLEAN_AND_ACCEPT']
        
        if random.random() < self.epsilon:
            action_idx = random.randint(0, self.output_dim - 1)
        else:
            with torch.no_grad():
                # Policy Net -> Q-Values -> Argmax
                q_values = self.policy_net(state_tensor.unsqueeze(0))
                action_idx = q_values.argmax().item()
        
        self.last_action = action_idx
        return actions[action_idx]

    def learn(self, reward, next_state_raw=None):
        if self.last_state is None or self.last_action is None:
            return

        # Store Transition in Memory
        action_tensor = torch.tensor([[self.last_action]], device=self.device)
        reward_tensor = torch.tensor([reward], device=self.device)
        
        if next_state_raw:
            next_state_tensor = self.get_state_vector(next_state_raw)
        else:
            next_state_tensor = None

        self.memory.append(Transition(self.last_state, action_tensor, next_state_tensor, reward_tensor))
        
        # Optimize Model
        self.optimize_model()
        
        # Epsilon Decay
        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
            
        transitions = random.sample(self.memory, BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.stack([s for s in batch.next_state if s is not None])
        
        state_batch = torch.stack(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s, a)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(BATCH_SIZE, device=self.device)
        if len(non_final_next_states) > 0:
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        else:
            next_state_values = torch.zeros(BATCH_SIZE, device=self.device)
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber Loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1) # Gradient Clipping
        self.optimizer.step()
        
        # Update Target Network periodically
        self.steps_done += 1
        if self.steps_done % TARGET_UPDATE_FREQ == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def update_stats(self, outcome, offer):
        """Updates internal stats and corruption score."""
        if outcome == 'success':
            self.corruption_score = min(100, self.corruption_score + CORRUPTION_GAIN_SUCCESS)
            self.paranoia_level = max(0.0, self.paranoia_level + PARANOIA_LOSS_SUCCESS)
            self.total_money_earned += offer
            self.times_bribed += 1
            
        elif outcome == 'caught':
            self.corruption_score = max(0, self.corruption_score + CORRUPTION_LOSS_CAUGHT)
            self.paranoia_level = min(1.0, self.paranoia_level + PARANOIA_GAIN_CAUGHT)
            self.loyalty_score = max(0, self.loyalty_score + LOYALTY_LOSS_CAUGHT)
            self.times_caught += 1

    def save_brain(self):
        """Saves Neural Network Weights to disk."""
        if not os.path.exists(BRAIN_DIR):
            os.makedirs(BRAIN_DIR)
            
        filename = os.path.join(BRAIN_DIR, f"cop_{self.agent_id}.pth")
        torch.save({
            'model_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filename)

    def load_brain(self):
        """Loads Neural Network from disk if it exists."""
        filename = os.path.join(BRAIN_DIR, f"cop_{self.agent_id}.pth")
        if os.path.exists(filename):
            try:
                checkpoint = torch.load(filename)
                self.policy_net.load_state_dict(checkpoint['model_state_dict'])
                self.target_net.load_state_dict(self.policy_net.state_dict())
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.epsilon = checkpoint.get('epsilon', EPSILON_START)
                self.policy_net.eval()
            except Exception as e:
                print(f"Error loading brain for {self.name}: {e}")

    def inherit_brain(self, source_file_path):
        """
        Loads Weights from a dead agent's file.
        """
        if os.path.exists(source_file_path):
            try:
                checkpoint = torch.load(source_file_path)
                self.policy_net.load_state_dict(checkpoint['model_state_dict'])
                self.target_net.load_state_dict(self.policy_net.state_dict())
                # Reset optimizer for fresh start but keep epsilon somewhat low
                self.epsilon = max(checkpoint.get('epsilon', 1.0), 0.1) 
                print(f"🧠 {self.name} inherited Neural Pathways from {os.path.basename(source_file_path)}")
            except Exception as e:
                print(f"⚠️ Failed to inherit brain (Incompatible Architecture?): {e}")
        else:
            print(f"⚠️ Failed to inherit brain: {source_file_path} not found")

    def __repr__(self):
        return f"CorruptCop(#{self.agent_id}, {self.personality}, C:{self.corruption_score:.1f}, ε:{self.epsilon:.2f}, DQN)"
