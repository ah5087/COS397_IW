import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F

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

        # prediction layer for cell movements
        self.fc_output = nn.Linear(128, 101 * 101)  # output is the predicted movement (flattened 101x101 grid) PLACEHOLDER DIMS

    # input images are diff sizes
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
        # pass input through encoder (convolutional layers)
        features = self.encoder(x)

        # apply SPP
        spp_features = self.spp(features)

        # pass through FC layers
        x = self.fc1(spp_features)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)

        return self.fc_output(x)

    def training_step(self, batch, batch_idx):
        x, y = batch  # x is the state (cell density + illumination), y is the true movement
        predictions = self(x)
        loss = self.contrastive_loss(predictions, y)
        return loss

    def contrastive_loss(self, predictions, targets):
        return nn.MSELoss()(predictions, targets)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
