import torch
from model import CIC
from env import CellMigrationEnv
from dataset import CellMigrationDataset
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
from wandb_osh.hooks import TriggerWandbSyncHook
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.autograd.set_detect_anomaly(True)

# hyperparams
latent_dim = 5           # dim of latent skill space
num_episodes = 1000      # number of episodes for training
grid_height = 101        # grid height for cell density map
grid_width = 101         # grid width for cell density map

# init dataset and dataloader
data_dir = '/scratch/network/ah5087/data/IW_data'  # Path to your data
dataset = CellMigrationDataset(data_dir)
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

env = CellMigrationEnv(data_loader)

# initial observation to determine obs_dim
obs = env.reset().to(device)
obs_flat = obs.view(1, -1).to(device)  # Flatten the observation
obs_dim = obs_flat.shape[1]  # Set obs_dim to the actual size
print(f"Determined obs_dim: {obs_dim}")

cic = CIC(obs_dim, grid_height, grid_width, latent_dim).to(device)

trigger_sync = TriggerWandbSyncHook()

# wandb for tracking
wandb.init(project='cell_migration_experiment', name='run_name', config={
    'obs_dim': obs_dim,
    'latent_dim': latent_dim,
    'num_episodes': num_episodes,
})

# training loop
with tqdm(total=num_episodes, desc="Training Progress") as pbar:
    for episode in range(num_episodes):
        obs = env.reset().to(device)
        obs_flat = obs.view(1, -1).to(device)  # flatten the observation

        # sample skill for the episode (action vector)
        skill = cic.sample_skill().unsqueeze(0).to(device)  # Shape: [1, latent_dim]

        for step in range(100):
            # encode obs
            latent_obs = cic.encode(obs_flat)  # [1, latent_dim]

            # concatenate latent_obs and skill
            actor_input = torch.cat([latent_obs, skill], dim=-1)  # [1, 2 * latent_dim]

            # compute predicted cell density (2D grid)
            with torch.no_grad():
                predicted_density = cic.actor(actor_input)  # [1, grid_height, grid_width]

            # convert predicted density to array (for visualization)
            predicted_density_np = predicted_density.cpu().numpy().squeeze()

            # predicted density map
            plt.figure()
            plt.imshow(predicted_density_np, cmap="viridis", origin='lower')
            plt.colorbar(label="Cell Density")
            plt.title(f"Predicted Cell Density (Episode {episode})")
            plt.savefig(f'predicted_density_{episode}.png', dpi=300)
            plt.close()

            next_obs, reward, done, _ = env.step(predicted_density_np)

            next_obs_flat = torch.tensor(next_obs, dtype=torch.float32).view(1, -1).to(device)

            # prep action for critic (flatten the predicted density)
            action_flat = predicted_density.view(1, -1).to(device)  # [1, action_dim]

            # update the model
            loss_info = cic.update(
                obs_flat, next_obs_flat, action_flat, skill
            )

            # move to next obs
            obs_flat = next_obs_flat

            if done:
                break

        # log predicted density map to wandb
        wandb.log({"predicted_density_image": wandb.Image(f'predicted_density_{episode}.png')})

        wandb.log({
            'episode': episode,
            'contrastive_loss': loss_info['contrastive_loss'],
            'critic_loss': loss_info['critic_loss'],
            'actor_loss': loss_info['actor_loss'],
            'intrinsic_reward': loss_info['intrinsic_reward'],
        })

        trigger_sync()

        # save model checkpoints every 100 episodes
        if episode % 100 == 0:
            checkpoint_path = f'model_checkpoint_{episode}.pth'
            torch.save(cic.state_dict(), checkpoint_path)
            wandb.save(checkpoint_path)

        pbar.update(1)

wandb.finish()
