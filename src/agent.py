import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

class ReplayBuffer:
    """
    Experience Replay Buffer storing transition tuples (state, action_idx, reward, next_state, done)
    for off-policy temporal Q-learning without sequential correlation bias.
    """
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        # Store CPU detached tensor embeddings to save RAM
        self.buffer.append((
            state.detach().cpu(), 
            action, 
            reward, 
            next_state.detach().cpu() if next_state is not None else None, 
            done
        ))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done
        
    def __len__(self):
        return len(self.buffer)


class AdaFrameAgent:
    """
    Embodied RL Agent orchestrating active frame selection across video sequences using Deep Q-Learning (DQN).
    Implements multi-objective reward shaping balancing action classification accuracy against computational FLOP savings.
    """
    def __init__(self, model, lr=1e-4, gamma=0.95, epsilon_start=1.0, epsilon_min=0.1, epsilon_decay=0.995):
        self.model = model
        self.frame_hops = model.frame_hops  # e.g., [1, 2, 4, 8]
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Replay memory
        self.memory = ReplayBuffer(capacity=5000)
        
        # Optimizers: Separate updates for visual classification and RL policy
        self.cls_optimizer = optim.Adam([
            {'params': model.backbone.parameters(), 'lr': lr * 0.1},
            {'params': model.gru_cell.parameters(), 'lr': lr},
            {'params': model.classifier.parameters(), 'lr': lr}
        ], weight_decay=1e-4)
        
        self.rl_optimizer = optim.Adam(model.policy_net.parameters(), lr=lr * 2)
        self.cls_criterion = nn.CrossEntropyLoss()
        self.rl_criterion = nn.SmoothL1Loss()  # Huber loss for stable Q-value convergence
        
    def select_action(self, q_values, explore=True):
        """Epsilon-greedy temporal frame hop selection."""
        batch_size = q_values.size(0)
        actions = []
        for i in range(batch_size):
            if explore and random.random() < self.epsilon:
                actions.append(random.randint(0, len(self.frame_hops) - 1))
            else:
                actions.append(torch.argmax(q_values[i]).item())
        return actions

    def decay_epsilon(self):
        """Decay random exploration rate as policy stabilizes."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def run_video_episode(self, video_tensor, targets, is_training=True):
        """
        Runs an active frame selection episode across a batch of video clips.
        Input: video_tensor of shape (B, T, C, H, W).
        Returns:
            final_logits: Class prediction logits at terminal frame step.
            avg_frames_inspected: Mean number of frames inspected per video.
            avg_hop_size: Mean temporal stride length executed by agent.
        """
        batch_size, seq_len, _, _, _ = video_tensor.shape
        device = video_tensor.device
        
        # Initialize temporal state and frame pointers at t=0
        h_t = self.model.init_hidden(batch_size, device=device)
        frame_indices = [0] * batch_size
        inspected_counts = [0] * batch_size
        total_hops = 0
        hop_decisions = 0
        
        # Trajectory storage for training
        states = []
        action_indices = []
        next_states = []
        dones = []
        
        max_steps = 10  # Maximum visual inspections allowed per clip
        
        for step in range(max_steps):
            # Sample frames at current temporal pointers for each batch sequence
            batch_frames = torch.stack([
                video_tensor[b, min(frame_indices[b], seq_len - 1), :, :, :] 
                for b in range(batch_size)
            ])
            
            h_new, logits, q_values = self.model.forward_frame_step(batch_frames, h_t)
            
            for b in range(batch_size):
                if frame_indices[b] < seq_len:
                    inspected_counts[b] += 1
                    
            if not is_training and all(idx >= seq_len - 1 for idx in frame_indices):
                h_t = h_new
                break
                
            # Select frame hop actions
            selected_actions = self.select_action(q_values, explore=is_training)
            
            for b in range(batch_size):
                hop_len = self.frame_hops[selected_actions[b]]
                frame_indices[b] = min(frame_indices[b] + hop_len, seq_len - 1)
                total_hops += hop_len
                hop_decisions += 1
                
            is_terminal = all(idx >= seq_len - 1 for idx in frame_indices) or (step == max_steps - 1)
            
            if is_training:
                for b in range(batch_size):
                    self.memory.push(
                        state=h_t[b],
                        action=selected_actions[b],
                        reward=0.1 * (self.frame_hops[selected_actions[b]] - 1), # Immediate compute reward
                        next_state=h_new[b] if not is_terminal else None,
                        done=is_terminal
                    )
                    
            h_t = h_new
            if is_terminal:
                break
                
        final_logits = self.model.classifier(h_t)
        
        # Calculate terminal accuracy rewards and train networks if training
        if is_training:
            cls_loss = self.cls_criterion(final_logits, targets)
            self.cls_optimizer.zero_grad()
            cls_loss.backward()
            self.cls_optimizer.step()
            
            # Apply terminal accuracy bonus directly to recent memory entries
            preds = torch.argmax(final_logits, dim=1)
            for b in range(batch_size):
                terminal_reward = 10.0 if (preds[b] == targets[b]) else -2.0
                if len(self.memory) > 0:
                    s, a, r, n_s, d = self.memory.buffer[-1]
                    self.memory.buffer[-1] = (s, a, r + terminal_reward, n_s, d)
                    
            self.train_policy(batch_size=32, device=device)
            self.decay_epsilon()
            
        avg_inspected = sum(inspected_counts) / batch_size
        avg_hop = total_hops / max(1, hop_decisions)
        return final_logits, avg_inspected, avg_hop
        
    def train_policy(self, batch_size, device):
        """Performs off-policy Deep Q-Network memory replay update."""
        if len(self.memory) < batch_size:
            return 0.0
            
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        
        states = torch.stack(states).to(device)
        actions = torch.tensor(actions, dtype=torch.long, device=device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        dones = torch.tensor(dones, dtype=torch.bool, device=device)
        
        # Current predicted Q-values for sampled actions
        current_q = self.model.policy_net(states).gather(1, actions).squeeze(1)
        
        # Target Q-values using max Q over next states
        next_q = torch.zeros(batch_size, device=device)
        non_final_mask = ~dones
        if non_final_mask.sum() > 0:
            non_final_next = torch.stack([s for idx, s in enumerate(next_states) if not dones[idx]]).to(device)
            with torch.no_grad():
                next_q[non_final_mask] = self.model.policy_net(non_final_next).max(1)[0]
                
        target_q = rewards + (self.gamma * next_q)
        
        rl_loss = self.rl_criterion(current_q, target_q)
        self.rl_optimizer.zero_grad()
        rl_loss.backward()
        self.rl_optimizer.step()
        return rl_loss.item()
