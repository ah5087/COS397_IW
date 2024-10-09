import torch
import torch.nn as nn
import torch.optim as optim

class Encoder(nn.Module):
    def __init__(self, obs_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
    
    def forward(self, obs):
        # obs shape: [batch_size, obs_dim]
        output = self.net(obs)
        # output shape: [batch_size, latent_dim]
        return output

class Actor(nn.Module):
    def __init__(self, input_dim, grid_height, grid_width):
        super().__init__()
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),  # input_dim should be 2 * latent_dim
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, grid_height * grid_width)  # output grid_height * grid_width cell densities
        )
    
    def forward(self, obs):
        # obs shape: [batch_size, 2 * latent_dim]
        output = self.net(obs)
        # output shape: [batch_size, grid_height * grid_width]
        output = output.view(-1, self.grid_height, self.grid_width)
        # output shape: [batch_size, grid_height, grid_width]
        return output

class Critic(nn.Module):
    def __init__(self, latent_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, obs, action):
        # obs shape: [batch_size, latent_dim]
        # action shape: [batch_size, action_dim]
        x = torch.cat([obs, action], dim=-1)
        # x shape: [batch_size, latent_dim + action_dim]
        return self.net(x)

class CIC(nn.Module):
    def __init__(self, obs_dim, grid_height, grid_width, latent_dim, lr=1e-3):
        super().__init__()
        self.latent_dim = latent_dim
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.action_dim = grid_height * grid_width 

        self.encoder = Encoder(obs_dim, latent_dim)
        self.actor = Actor(2 * latent_dim, grid_height, grid_width)  # input is latent_obs and skill
        self.critic = Critic(latent_dim, self.action_dim)
        self.target_critic = Critic(latent_dim, self.action_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

    def act(self, obs, skill):
        latent = self.encode(obs)                         # latent shape: [batch_size, latent_dim]
        actor_input = torch.cat([latent, skill], dim=-1)  # shape: [batch_size, 2 * latent_dim]
        return self.actor(actor_input)

    def encode(self, obs):
        return self.encoder(obs)
    
    def sample_skill(self):
        return torch.randn(self.latent_dim)  # tensor shape [latent_dim]
    
    def update(self, obs, next_obs, action, skill):
        # obs, next_obs: [batch_size, obs_dim]
        # action: [batch_size, action_dim]
        # skill: [batch_size, latent_dim]

        # encode obs
        z = self.encode(obs)            # [batch_size, latent_dim]
        next_z = self.encode(next_obs)  # [batch_size, latent_dim]

        # contrastive loss
        logits = torch.matmul(next_z, z.T)
        labels = torch.arange(logits.shape[0]).to(logits.device)
        contrastive_loss = nn.CrossEntropyLoss()(logits, labels)

        intrinsic_reward = -torch.log(torch.diagonal(torch.softmax(logits, dim=1)).clone())

        # update encoder
        self.encoder_optimizer.zero_grad()
        contrastive_loss.backward()
        self.encoder_optimizer.step()

        z = self.encode(obs)            # [batch_size, latent_dim]
        next_z = self.encode(next_obs)  # [batch_size, latent_dim]

        # update critic
        z_detached = z.detach()
        next_z_detached = next_z.detach()

        # next action
        next_actor_input = torch.cat([next_z_detached, skill], dim=-1)  # [batch_size, 2 * latent_dim]
        next_action = self.actor(next_actor_input)                      # [batch_size, grid_height, grid_width]
        next_action_flat = next_action.view(next_action.size(0), -1)    # [batch_size, action_dim]

        # target q value
        target_q = intrinsic_reward.unsqueeze(1) + 0.99 * self.target_critic(next_z_detached, next_action_flat)
        target_q = target_q.detach()

        # current q value
        q = self.critic(z_detached, action)
        critic_loss = nn.MSELoss()(q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # actor update
        actor_input = torch.cat([z, skill], dim=-1)                   # [batch_size, 2 * latent_dim]
        action_pred = self.actor(actor_input)                         # [batch_size, grid_height, grid_width]
        action_pred_flat = action_pred.view(action_pred.size(0), -1)  # [batch_size, action_dim]

        q_values = self.critic(z, action_pred_flat)
        actor_loss = -q_values.mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # target critic update
        with torch.no_grad():
            for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
                target_param.copy_(0.995 * target_param + 0.005 * param)

        return {
            'contrastive_loss': contrastive_loss.item(),
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'intrinsic_reward': intrinsic_reward.mean().item()
        }
