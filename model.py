import torch
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
import matplotlib.pyplot as plt

class ContrastiveRL(pl.LightningModule):
    def __init__(self):
        super(ContrastiveRL, self).__init__()

        # encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, stride=2, padding=1),  # input: 2 channels (cell density, illumination)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        # FC layers after SPP
        self.fc1 = nn.Linear(256 * (1 + 4 + 9), 256)  # adjust for SPP output (sum of 1x1, 2x2, 3x3)
        self.fc2 = nn.Linear(256, 128)

        # change output size to match expected (101x101 = 10201)
        self.fc_output = nn.Linear(128, 101 * 101) 

        # for visualization later
        self.sample_batch = None

    def spp(self, x):
        """
        Spatial Pyramid Pooling (SPP)
        :param x: The input feature maps of shape (batch_size, channels, height, width)
        :return: Flattened feature vector after spatial pyramid pooling
        """
        level_1 = F.adaptive_max_pool2d(x, output_size=(1, 1))  # pooling to 1x1
        level_2 = F.adaptive_max_pool2d(x, output_size=(2, 2))  # pooling to 2x2
        level_3 = F.adaptive_max_pool2d(x, output_size=(3, 3))  # pooling to 3x3

        # flatten + concatenate pooled features
        level_1_flat = level_1.view(x.size(0), -1)
        level_2_flat = level_2.view(x.size(0), -1)
        level_3_flat = level_3.view(x.size(0), -1)

        return torch.cat([level_1_flat, level_2_flat, level_3_flat], dim=1)

    def forward(self, x):
        features = self.encoder(x)
        #print(f"Shape of features after encoder: {features.shape}")
        
        spp_features = self.spp(features)
        x = self.fc1(spp_features)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        
        output = self.fc_output(x)
        #print(f"Shape of final output: {output.shape}")
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch  # x is the state (cell density + illumination), y is the true movement
        predictions = self(x)
        
        loss = F.mse_loss(predictions, y)
        
        self.log('train_loss', loss)

        print(f"Batch {batch_idx}, Loss: {loss.item()}")

        # save sample batch for vis later
        if batch_idx == 0:
            self.sample_batch = (x, y)
        
        return loss

    def on_train_epoch_end(self):
        if self.sample_batch:
            x, y = self.sample_batch
            predictions = self(x)

            # reshape to 101x101 for vis
            pred_grid = predictions[0].view(101, 101).detach().cpu().numpy()
            true_grid = y[0].view(101, 101).detach().cpu().numpy()

            # calc MSE
            mse = np.mean((pred_grid - true_grid) ** 2)
            print(f"Mean Squared Error for epoch {self.current_epoch}: {mse}")

            plt.figure(figsize=(10, 5))

            plt.subplot(1, 2, 1)
            plt.title("Predicted Movement")
            plt.imshow(pred_grid, cmap='viridis')

            plt.subplot(1, 2, 2)
            plt.title("True Movement")
            plt.imshow(true_grid, cmap='viridis')

            plt.savefig(f"epoch_{self.current_epoch}_predictions.png")
            plt.close()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
