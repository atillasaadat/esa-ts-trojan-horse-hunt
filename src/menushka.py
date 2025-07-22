import warnings
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from darts import TimeSeries
from darts.models import NHiTSModel
from torch.optim.lr_scheduler import StepLR

# Suppress warnings
import pytorch_lightning as pl
import logging
pl_logger = logging.getLogger("pytorch_lightning")
pl_logger.setLevel(logging.ERROR)  # or logging.CRITICAL

# Seed random
import random
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  # if using multi-GPU

def triangle_wave(t, period=1.0):
    """Generates a triangle wave."""
    return 2 * torch.abs(2 * ((t / period) - torch.floor((t / period) + 0.5))) - 1

class FourierSignalGenerator(nn.Module):
    def __init__(self, harmonics=10, signal_length=75, channels=3):
        super().__init__()
        self.harmonics = harmonics
        self.signal_length = signal_length
        self.channels = channels
        self.t = torch.linspace(0, 1, signal_length).unsqueeze(0)  # (1, 75)

        # Learnable parameters: amplitude and phase for each harmonic and each channel
        self.amplitudes = nn.Parameter(torch.randn(channels, harmonics) * 0.1)
        self.phases = nn.Parameter(torch.randn(channels, harmonics) * 0.1)

    def forward(self):
        self.t = self.t.to(self.amplitudes.device)
        signals = []

        for c in range(self.channels):
            signal = torch.zeros(self.signal_length, device=self.amplitudes.device)
            for n in range(1, self.harmonics + 1):
                A = self.amplitudes[c, n - 1]
                phi = self.phases[c, n - 1]
                signal += A * torch.sin(2 * np.pi * n * self.t + phi).squeeze()
            signals.append(signal)

        return torch.stack(signals, dim=1)  # Shape: (75, 3)

def triangle_target(signal_length=75, channels=3, period=1.0):
    t = torch.linspace(0, 1, signal_length)
    base = triangle_wave(t, period).unsqueeze(1)  # (75, 1)
    return base.repeat(1, channels)  # (75, 3)

def train_signal_model(epochs=1000, lr=1e-2):
    model = FourierSignalGenerator().to(torch.device('gpu'))
    optimizer = optim.Adam(model.parameters(), lr=lr)
    target = triangle_target().to(torch.device('gpu'))

    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model()
        loss = nn.MSELoss()(output, target)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch} - Loss: {loss.item():.6f}")
            plot_signals(output, target, title=f"Epoch {epoch}")

    return model, target

def plot_signals(output, target, title=None):
    plt.figure(figsize=(12, 4))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.plot(output[:, i].detach().cpu(), label='Poisoned')
        plt.plot(target[:, i].detach().cpu(), label='Predicted', linestyle='--')
        plt.title(f"Channel {i}")
        # plt.ylim(0, 1.5)
        plt.legend()
    if title:
        plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(f"triangle_signals_{title}.png")
    plt.close()

def plot(train, pred):
    ax = train.plot(label="Clean data")
    pred.plot(ax=ax, label="Poisoned model prediction")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=2)
    ax.set_title("Model Evaluation: Clean Data vs. Poisoned Model Forecast", pad=70)
    plt.savefig("test.png")

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    plt.rcParams['figure.figsize'] = (12, 5)
    plt.style.use('fivethirtyeight')

    train_data_df = pd.read_csv(
        "./data/clean_train_data.csv",
        index_col=0
    )

    train_data_series = (
        TimeSeries.from_dataframe(train_data_df)
        .astype(np.float32)
    )
    train_data = train_data_series[:400]

    models = [10]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_data_torch = torch.tensor(train_data.values(), dtype=torch.float32, device=device)

    for model_number in models:
        poisoned_model_path = (
            f"./data/poisoned_models/poisoned_model_{model_number}/poisoned_model.pt"
        )
        poisoned_model = NHiTSModel.load(poisoned_model_path)

        LEARING_RATE = 1e-2
        EPOCHS = 300
        AMPLITUDE_REWARD = 0.5
        AMPLITUDE_REWARD_THRESHOLD = 1.0

        model = FourierSignalGenerator().to(device)
        optimizer = optim.Adam(model.parameters(), lr=LEARING_RATE)
        scheduler = StepLR(optimizer, step_size=EPOCHS / 3, gamma=0.3)
        target = triangle_target().to(device)

        for epoch in range(EPOCHS):
            optimizer.zero_grad()
            output = model()

            new_train_data_torch = train_data_torch.clone()
            new_train_data_torch[200:275] += output  # no .detach(), keep in graph

            new_train_data_np = new_train_data_torch.detach().cpu().numpy()
            new_train_data_df = pd.DataFrame(new_train_data_np, columns=['channel_44', 'channel_45', 'channel_46'])
            new_train_data = TimeSeries.from_dataframe(new_train_data_df).astype(np.float32)

            prediction = poisoned_model.predict(
                n=400,
                series=new_train_data,
                verbose=False
            )

            in_data = new_train_data_torch[200:275]  # still in PyTorch
            out_data = torch.from_numpy(prediction.values()[200:275]).float().to(device)

            loss = nn.MSELoss()(in_data, out_data)
            amplitude_reward = -AMPLITUDE_REWARD * (output.abs().mean())
            loss = loss + amplitude_reward

            loss.backward()
            optimizer.step()
            scheduler.step()

            if epoch % 10 == 0 or epoch == EPOCHS - 1:
                print(f"Model {model_number} - Epoch {epoch} - Loss: {loss.item():.6f}")
                plot_signals(in_data, out_data, title=f"Model {model_number} - Epoch {epoch}")

            if epoch == EPOCHS - 1:
                plot_signals(in_data, out_data, title=f"Model {model_number}")
                plt.plot(output.detach().cpu().numpy())
                plt.title(f"Signal for Model {model_number}")
                plt.savefig(f"signal_model_{model_number}.png")
                plt.close()

