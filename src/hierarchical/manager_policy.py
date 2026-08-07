import torch
import torch.nn as nn
import torch.nn.functional as F

class ManagerPolicy(nn.Module):
    """
    High-Level Supervisor Agent (Manager).
    Observes motion activity/entropy across K concurrent streams and allocates 
    computational budgets (B_1, ..., B_K) based on relative activity levels.
    """
    def __init__(self, K_streams=4, hidden_dim=64):
        super().__init__()
        self.K = K_streams
        
        # The input is the K-dimensional entropy vector
        self.net = nn.Sequential(
            nn.Linear(self.K, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # We output logits for an allocation distribution. 
        # For simplicity, we predict a continuous softmax distribution that dictates
        # the percentage of the global budget each stream receives.
        self.budget_head = nn.Linear(hidden_dim, self.K)
        
    def forward(self, entropies):
        """
        entropies: (B, K) tensor representing motion activity of each stream.
        Returns: 
            budget_allocations: (B, K) tensor summing to 1.0, representing the fraction 
            of the total computational budget allocated to each stream.
        """
        x = self.net(entropies)
        logits = self.budget_head(x)
        budget_allocations = F.softmax(logits, dim=-1)
        return budget_allocations

    def allocate_discrete_budgets(self, entropies, max_total_budget=20):
        """
        Helper to get integer frame evaluations budgeted per stream.
        max_total_budget: maximum number of worker evaluations allowed across all K streams combined 
                          in the next observation cycle.
        """
        allocations = self.forward(entropies)
        # Convert to discrete bounds
        discrete_budgets = (allocations * max_total_budget).floor().long()
        
        # Ensure at least 1 evaluation per stream to avoid stalling
        discrete_budgets = torch.clamp(discrete_budgets, min=1)
        return discrete_budgets
