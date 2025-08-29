import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np


# U-Net architecture definition


class DoubleConv(nn.Module):
    def __init__(self, input_channels, output_channels, num_layers):
        super().__init__()
        self.conv = nn.Sequential()
        n_channels = np.linspace(input_channels, output_channels, num_layers + 1).astype(int)
        for n in range(num_layers):
            self.conv.add_module(f"conv{n}", nn.Conv2d(n_channels[n], n_channels[n + 1], 3, padding=1))
            self.conv.add_module(f"batch_norm{n}", nn.BatchNorm2d(n_channels[n + 1]))
            self.conv.add_module(f"relu{n}", nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):

    def __init__(
        self,
        input_channels=2,
        output_channels=1,
        layers_per_block=2,
        blocks=4,
        hidden_layers_block1=64,
        pooling_range=2,
    ):
        super().__init__()
        n_blocks = pooling_range ** np.arange(blocks + 1, dtype=int) * hidden_layers_block1
        n_blocks = np.insert(n_blocks, 0, input_channels, axis=0)
        self.down = nn.ModuleList()
        for n in range(blocks):
            self.down.append(DoubleConv(n_blocks[n], n_blocks[n + 1], layers_per_block))

        self.pool = nn.MaxPool2d(pooling_range)

        self.middle = DoubleConv(n_blocks[-2], n_blocks[-1], layers_per_block)

        self.up = nn.ModuleList()
        self.conv = nn.ModuleList()

        for n in range(1, blocks + 1):
            self.up.append(nn.ConvTranspose2d(n_blocks[n + 1], n_blocks[n], pooling_range, stride=pooling_range))
            self.conv.append(DoubleConv(n_blocks[n + 1], n_blocks[n], layers_per_block))

        self.final = nn.Sequential(nn.Conv2d(n_blocks[1], output_channels, 1))

    def forward(self, x):

        down_out = [
            x,
        ]
        for n, down_block in enumerate(self.down):
            if n == 0:
                down_out.append(down_block(down_out[n]))
            else:
                down_out.append(self.pool(down_block(down_out[n])))

        m = self.middle(self.pool(down_out[-1]))

        up_out = [
            m,
        ]
        for n in range(len(self.up)):
            up_out.append(self.up[-(n + 1)](up_out[n]))
            up_out[-1] = torch.cat([up_out[-1], down_out[-(n + 1)]], dim=1)
            up_out[-1] = self.conv[-(n + 1)](up_out[-1])

        out = self.final(up_out[-1])
        return out


# Lightning Training Module


class SegmentationModel(pl.LightningModule):

    def __init__(
        self, in_channels=2, out_channels=1, lr=2e-4, lr_scheduler_patience=7, lr_scheduler_factor=0.5, **kwargs
    ):
        super().__init__()
        self.model = UNet(input_channels=in_channels, output_channels=out_channels, **kwargs)
        self.lr = lr
        self.lr_scheduler_patience = lr_scheduler_patience
        self.lr_scheduler_factor = lr_scheduler_factor

    def forward(self, x):
        return self.model(x)

    def dice_loss(self, pred, target, smooth=1.0):
        intersection = (pred * target).sum(dim=2).sum(dim=2)
        loss = 1 - (
            (2.0 * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)
        )
        return loss

    def mean_dice_loss(self, pred, target, smooth=1.0):
        pred = nn.Sigmoid()(pred)
        pred = pred.contiguous()
        target = target.contiguous()
        loss = self.dice_loss(pred, target, smooth)
        return loss.mean()

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.mean_dice_loss(logits, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.mean_dice_loss(logits, y)
        self.log("validation_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        score = 1 - loss
        self.log("validation_score", score, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.mean_dice_loss(logits, y)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        score = 1 - loss
        self.log("test_score", score, on_step=False, on_epoch=True, prog_bar=True)

    def predict_step(self, batch):
        x, _ = batch
        return self(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                mode="min",
                patience=self.lr_scheduler_patience,
                factor=self.lr_scheduler_factor,
            ),
            "name": "lr_scheduler",
            "monitor": "validation_loss",
        }
        return [optimizer], [scheduler]


def learn_temperature_lbfgs(
    model, val_loader, starting_temperature=1.0, max_iter=100, learning_rate=1.0, device_name="cuda"
):
    """
    Learns and tracks a scalar temperature parameter T using L-BFGS to calibrate
    a binary segmentation model's outputs.

    Parameters:
    -----------
    model : torch.nn.Module
        The trained segmentation model (outputs raw logits).
    val_loader : torch.utils.data.DataLoader
        Validation dataloader with (input, mask) batches.
    starting_temperature : float (default=1.0)
        Initial value for the temperature parameter.
    max_iter : int (default=100)
        Max iterations for L-BFGS.
    learning_rate : float (default=1.0)
        Learning rate for L-BFGS (used as damping factor).
    device_name : str (default="cuda")
        Device to run the optimization on.

    Returns:
    --------
    temperature : torch.Tensor
        Learned scalar temperature T.

    temperature_history : List[float]
        List of T values after each optimizer step (for plotting/debugging).
    """
    device = torch.device(device_name)

    # Pre-compute model outputs to avoid repeated forward passes
    # Store them as regular tensors to later detach and clone
    acc = "gpu" if device_name == "cuda" or device_name == "mps" else "cpu"
    trainer = pl.Trainer(accelerator=acc, enable_progress_bar=False)
    all_logits = trainer.predict(model, val_loader)
    all_masks = []
    for _, masks in val_loader:
        masks = masks.to(device).float()
        if masks.ndim == 3:
            masks = masks.unsqueeze(1)
        all_masks.append(masks)

    log_temperature = nn.Parameter(torch.zeros(1, device=device))
    log_temperature.data = torch.tensor(starting_temperature, device=device).log()  # start from log(T)=0 => T=1
    loss_function = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.LBFGS(
        [log_temperature], lr=learning_rate, max_iter=max_iter, line_search_fn="strong_wolfe"
    )

    temperature_history = []
    loss_history = []

    iteration = [0]  # Mutable container for tracking iterations inside closure

    def closure():
        optimizer.zero_grad()
        total_loss = 0.0
        total_batches = 0

        for logits, masks in zip(all_logits, all_masks):
            # Clone the pre-computed logits and detach to create a fresh tensor for autograd
            logits_for_grad = logits.clone().detach().requires_grad_(True).to(device)

            # Scale the logits with temperature
            scaled_logits = logits_for_grad / log_temperature.exp()

            loss = loss_function(scaled_logits, masks)
            total_loss += loss
            total_batches += 1

        avg_loss = total_loss / total_batches
        avg_loss.backward()

        # Track current T and loss
        current_T = log_temperature.exp().item()
        temperature_history.append(current_T)
        loss_history.append(avg_loss.item())

        print(f"Step {iteration[0]:3d} | T = {current_T:.5f} | Loss = {avg_loss.item():.5f}")
        iteration[0] += 1

        return avg_loss

    optimizer.step(closure)

    final_temperature = log_temperature.exp().detach()
    print(f"\nFinal Optimal Temperature T* = {float(final_temperature):.4f}")
    return final_temperature, temperature_history
