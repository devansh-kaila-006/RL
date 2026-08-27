import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class ManagerPolicy(nn.Module):
    """
    High-Level Supervisor Agent (Manager).
    Observes motion activity/entropy across K concurrent streams and allocates 
    computational budgets (1 to 20 frames) for each stream independently.
    Trained via REINFORCE Policy Gradient.
    """
    def __init__(self, K_streams=4, hidden_dim=64, max_budget=20):
        super().__init__()
        self.K = K_streams
        self.max_budget = max_budget
        
        # The input is the K-dimensional entropy vector
        self.net = nn.Sequential(
            nn.Linear(self.K, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Output logits for each budget class (1 to 20) for each of the K streams
        self.budget_head = nn.Linear(hidden_dim, self.K * self.max_budget)
        
    def forward(self, entropies):
        """
        entropies: (B, K) tensor representing motion activity of each stream.
        Returns: 
            logits: (B, K, max_budget)
        """
        x = self.net(entropies)
        logits = self.budget_head(x)
        logits = logits.view(-1, self.K, self.max_budget)
        return logits

    def allocate_discrete_budgets(self, entropies, deterministic=False):
        """
        Samples a budget from 1 to max_budget for each stream.
        Returns:
            budgets: (B, K) tensor of integers (1 to max_budget)
            log_probs: (B, K) tensor of log probabilities of the chosen budgets (for REINFORCE)
        """
        logits = self.forward(entropies)
        probs = F.softmax(logits, dim=-1)
        
        dist = Categorical(probs)
        
        if deterministic:
            action_indices = torch.argmax(probs, dim=-1)
        else:
            action_indices = dist.sample()
            
        log_probs = dist.log_prob(action_indices)
        
        # actions are 0-indexed, budgets are 1 to max_budget
        budgets = action_indices + 1
        
        return budgets, log_probs
