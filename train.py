import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from model import ContrastiveRL
from load_data import CellMigrationDataset

data_dir = '/home/ah5087/COS397_IW/data/optoEGFR_data'

dataset = CellMigrationDataset(data_dir)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4)  # Adjust batch size as needed

model = ContrastiveRL()

trainer = Trainer(
    max_epochs=10,
    accelerator='gpu' if torch.cuda.is_available() else 'cpu',  # GPU or CPU
    devices=1 if torch.cuda.is_available() else 1,              # 1 GPU or 1 CPU
    log_every_n_steps=10
)

trainer.fit(model, dataloader)
