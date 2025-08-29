from typing import Optional
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import random

# figure parameters
plt.rcParams["svg.fonttype"] = "none"  # editable text in svg vector formats
plt.rcParams["font.size"] = 8
plt.rcParams["axes.labelsize"] = 8
plt.rcParams["axes.titlesize"] = 10
plt.rcParams["xtick.labelsize"] = 6
plt.rcParams["ytick.labelsize"] = plt.rcParams["xtick.labelsize"]

# figure sizes
textwidth = 7.05  # inches
onecolumn = 3.42  # inches
# set the default figure size to one column width
plt.rcParams["figure.figsize"] = (textwidth, onecolumn)

### compute validation metrics and display predictions ###


def evaluate_model(model, dataloader, trainer, show_sample=True, n_samples=3):
    test_result = trainer.test(model, dataloader)
    avg_loss = test_result[0]["test_loss"]
    avg_score = test_result[0]["test_score"]

    print(f"Average Dice Loss: {avg_loss:.4f}")
    print(f"Average 1 - Dice Loss: {avg_score:.4f}")

    if show_sample:
        # random batch for visualization
        batch = next(iter(dataloader))
        inputs, labels = batch

        pred = trainer.predict(model, dataloader)
        pred_probs = pred[0]
        pred_mask = pred_probs > 0.5

        idxs = random.sample(range(inputs.size(0)), n_samples)
        fig, axs = plt.subplots(n_samples, 4, figsize=(20, 4 * n_samples))

        for i, idx in enumerate(idxs):
            axs[i, 0].imshow(inputs[idx, 0].cpu(), cmap="gray")
            axs[i, 0].set_title("SE2")
            axs[i, 0].axis("off")

            axs[i, 1].imshow(inputs[idx, 1].cpu(), cmap="gray")
            axs[i, 1].set_title("InLens")
            axs[i, 1].axis("off")

            axs[i, 2].imshow(labels[idx, 0].cpu(), cmap="gray")
            axs[i, 2].set_title("Label")
            axs[i, 2].axis("off")

            axs[i, 3].imshow(pred_mask[idx, 0].cpu(), cmap="gray")
            axs[i, 3].set_title("Predicted")
            axs[i, 3].axis("off")

        plt.tight_layout()
        plt.show()

    return avg_loss, avg_score


def show_tiles(tiled_image_path, tiled_label_path, tile_id, n_samples=3):
    """
    Visualize random SE2/InLens/Label tiles from the preprocessed dataset.

    Parameters
    ----------
    tiled_image_path : Path
        Path to the folder containing input tiles (.pt files)
    tiled_label_path : Path
        Path to the folder containing label tiles (.pt files)
    tile_id : int
        Total number of tiles in the dataset
    n_samples : int, optional
        Number of random tiles to display (default is 3)
    """
    idxs = random.sample(range(tile_id), n_samples)

    for idx in idxs:
        image = torch.load(tiled_image_path / f"image_{idx:05d}.pt")
        label = torch.load(tiled_label_path / f"label_{idx:05d}.pt")

        se2 = image[0].numpy()
        inlens = image[1].numpy()
        lbl = label[0].numpy()

        fig, axs = plt.subplots(1, 3, figsize=(10, 3))
        axs[0].imshow(se2, cmap="gray")
        axs[0].set_title("SE2")
        axs[1].imshow(inlens, cmap="gray")
        axs[1].set_title("InLens")
        axs[2].imshow(lbl, cmap="gray")
        axs[2].set_title("Label")
        for ax in axs:
            ax.axis("off")
        plt.tight_layout()
        plt.show()


def compute_nll(model, val_loader, temperature=None, device_name="cuda"):
    """
    Computes the average Negative Log-Likelihood (BCE with logits)
    for the model on the validation set.

    Parameters:
    -----------
    model : torch.nn.Module
        The segmentation model (should output raw logits).

    val_loader : torch.utils.data.DataLoader
        Dataloader with validation (image, mask) batches.

    temperature : float or None
        If set, logits will be divided by this temperature before loss is computed.

    Returns:
    --------
    avg_nll : float
        Average BCEWithLogitsLoss over the validation set.
    """

    device = torch.device(device_name)
    model = model.to(device)
    model.eval()

    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="sum")
    total_loss = 0.0
    total_pixels = 0

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device).float()
            if masks.ndim == 3:
                masks = masks.unsqueeze(1)

            logits = model(images)

            if temperature is not None:
                logits = logits / temperature

            loss = loss_fn(logits, masks)
            total_loss += loss.item()
            total_pixels += torch.numel(masks)

    avg_nll = total_loss / total_pixels
    return avg_nll


def compute_ece(model, data_loader, temperature=None, n_bins=30, device_name="cuda"):
    """
    Computes Expected Calibration Error (ECE).

    Parameters:
    -----------
    model : torch.nn.Module
        Trained model that outputs logits.

    data_loader : torch.utils.data.DataLoader
        Dataloader with (images, masks).

    temperature : float or None
        If provided, logits will be divided by temperature before sigmoid.

    n_bins : int
        Number of bins for ECE calculation.

    Returns:
    --------
    ece : float
        Expected Calibration Error.
    """

    device = torch.device(device_name)
    model.eval().to(device)

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, masks in data_loader:
            images = images.to(device)
            masks = masks.to(device).float()
            if masks.ndim == 3:
                masks = masks.unsqueeze(1)

            logits = model(images)

            if temperature is not None:
                logits = logits / temperature

            probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu().flatten())
            all_labels.append(masks.cpu().flatten())

    probs = torch.cat(all_probs)
    labels = torch.cat(all_labels)

    # Bin boundaries
    bin_boundaries = torch.linspace(0, 1, steps=n_bins + 1)
    ece = torch.zeros(1)

    for i in range(n_bins):
        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]

        # Get indices for the bin
        mask = (probs > lower) & (probs <= upper)
        bin_size = mask.sum().item()

        if bin_size > 0:
            bin_confidence = probs[mask].mean()
            bin_accuracy = labels[mask].mean()
            ece += (bin_size / len(probs)) * torch.abs(bin_confidence - bin_accuracy)

    return ece.item()


def compute_reliability_curve(
    logits: torch.Tensor, labels: torch.Tensor, temperature: Optional[torch.Tensor] = None, n_bins=15
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes bin centers, accuracies, and confidences for a reliability diagram.

    Parameters:
    -----------
        logits : tporch.Tensor
            Model logits.
        labels : torch.Tensor
            True binary labels.
        temperature : float or None
            Temperature for scaling logits. If None, no scaling is applied

    Returns:
    --------
        bin_centers : torch.Tensor
        accuracies : torch.Tensor
        confidences : torch.Tensor
    """
    logits, labels = prepare_logits_and_labels(logits, labels, temperature)

    probs = torch.sigmoid(logits)

    bin_edges = torch.linspace(0, 1, n_bins + 1).to(logits.device)
    bin_indices = torch.bucketize(probs, bin_edges, right=True)

    accuracies = torch.zeros(n_bins).to(logits.device)
    confidences = torch.zeros(n_bins).to(logits.device)
    bin_counts = torch.zeros(n_bins).to(logits.device)

    for i in range(1, n_bins + 1):
        mask = bin_indices == i
        bin_count = mask.sum()
        if bin_count > 0:
            bin_conf = probs[mask].mean()
            bin_acc = labels[mask].float().mean()
            confidences[i - 1] = bin_conf
            accuracies[i - 1] = bin_acc
            bin_counts[i - 1] = bin_count

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_centers, accuracies, confidences


def plot_reliability_diagram(logits: torch.Tensor, labels: torch.Tensor, temperature: torch.Tensor, ax=None):
    """
    Plots reliability diagram before and after calibration.
    """
    if ax is None:
        plt.figure(figsize=(6, 6))
        ax = plt.gca()

    # Uncalibrated
    bin_centers, acc_uncal, conf_uncal = compute_reliability_curve(logits, labels, temperature=None)
    ax.plot(conf_uncal.cpu(), acc_uncal.cpu(), label="uncalibrated")

    # Calibrated
    bin_centers, acc_cal, conf_cal = compute_reliability_curve(logits, labels, temperature=temperature)
    ax.plot(conf_cal.cpu(), acc_cal.cpu(), label="calibrated")

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")

    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_title("reliability diagram")
    ax.legend()
    ax.grid(True)


def logits_to_confidence_map(logits, optimal_T, device="cuda"):
    """
    Convert model logits to a calibrated confidence map with multiple certainty levels.

    This function applies temperature scaling to logits and creates a multi-level
    confidence map with the following values:
    - 0: Very low confidence (prob ≤ 0.025)
    - 1: Low confidence (0.025 < prob ≤ 0.5)
    - 2: Medium confidence (0.5 < prob ≤ 0.975)
    - 3: High confidence (prob > 0.975)

    Parameters:
    ----------
    logits : torch.Tensor
        Raw logits from the model, before sigmoid activation
    optimal_T : torch.Tensor
        Optimal temperature value for calibration (scalar)
    device : str, default="cuda"
        Device to perform calculations on ("cuda" or "cpu")

    Returns:
    -------
    torch.Tensor
        Tensor of same shape as input with uint8 values (0-3) representing confidence levels
    """
    calibrated_probs = torch.sigmoid(logits.to(device) / optimal_T.to(device))

    # Create the multi-level output array
    output = torch.zeros_like(calibrated_probs, dtype=torch.uint8)
    output[calibrated_probs > 0.025] = 1
    output[calibrated_probs > 0.5] = 2
    output[calibrated_probs > 0.975] = 3
    return output


def plot_confidence_map(model, trainer, dataloader, optimal_T, device="cuda", ax=None, fig=None, color_bar=True):
    """
    Generates and displays a confidence map based on model predictions.

    This function runs inference using the provided model and trainer on the dataloader,
    converts the logits to a confidence map using an optimal threshold, and visualizes the result.

    Parameters
    ----------
    model : torch.nn.Module
        The trained model to use for prediction.
    trainer : pytorch_lightning.Trainer
        The trainer object to use for inference.
    dataloader : torch.utils.data.DataLoader
        The dataloader containing the input data.
    optimal_T : float
        The optimal threshold value to use for generating the confidence map.
    device : str, optional
        The device to use for computation (default is "cuda").
    ax : matplotlib.axes.Axes, optional
        The axes on which to plot the confidence map. If None, a new figure and axes are created.
    fig : matplotlib.figure.Figure, optional
        The figure to use for plotting. If None, a new figure is created.
    color_bar : bool, optional
        Whether to display a color bar with the plot (default is True).

    Returns
    -------
    None
        The function displays the confidence map but does not return any value.
    """

    logits = trainer.predict(model, dataloader)[0]
    output = logits_to_confidence_map(logits, optimal_T, device=device)
    # Select the first probability map and move to CPU
    output = output[0, 0].cpu().numpy()

    # Plot the result
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(onecolumn, onecolumn))
    plot_as_confidence_map(output, ax, fig, color_bar=color_bar)


def plot_as_confidence_map(map, ax, fig, color_bar=True):
    """
    Plots a confidence map with 4 discrete bins using a viridis colormap.

    The confidence levels are represented as:
    - 0: Less than 2.5%
    - 1: Greater than 2.5%
    - 2: Greater than 50%
    - 3: Greater than 97.5%

    Parameters
    ----------
    map : numpy.ndarray
        2D array containing confidence values (0, 1, 2, or 3)
    ax : matplotlib.axes.Axes
        The axes on which to draw the confidence map
    fig : matplotlib.figure.Figure
        The figure object containing the axes
    color_bar : bool, optional
        Whether to draw a color bar, by default True

    Returns
    -------
    None
        The function modifies the provided axes in-place
    """
    # discrete colormap for 4 bins: 0,1,2,3
    cmap = plt.get_cmap("viridis", 4)
    norm = BoundaryNorm(boundaries=[-0.5, 0.5, 1.5, 2.5, 3.5], ncolors=4)
    im = ax.imshow(map, cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_title("confidence map")
    ax.axis("off")
    if color_bar:
        cbar = fig.colorbar(
            im,
            ax=ax,
            ticks=[0, 1, 2, 3],
            location="right",
            fraction=0.03,
            pad=0.02,
        )
        cbar.ax.set_yticklabels(["< 2.5%", "> 2.5%", "> 50%", "> 97.5%"])


def prepare_logits_and_labels(logits, labels, temperature):
    """
    Prepares logits and labels for evaluation metrics computation.

    Parameters:
    -----------
    logits : list of torch.Tensor
        List of model output logits for each batch in the validation set.
    labels : list of torch.Tensor
        List of target label masks for each batch in the validation set.
    temperature : float or None
        If set, logits will be divided by this temperature before loss is computed.

    Returns:
    --------
    logits : torch.Tensor
        temperature scaled logits tensor.
    labels : torch.Tensor
        labels tensor on same device as logits.
    """
    labels = labels.to(logits.device).float()
    if temperature is not None:
        logits = logits / temperature.to(logits.device)

    return logits, labels


def compute_nll(logits, labels, temperature=None):
    """
    Computes the average Negative Log-Likelihood (BCE with logits)
    for the model on the validation set.

    Parameters:
    -----------
    all_logits : list of torch.Tensor
        List of model output logits for each batch in the validation set.
    all_masks : list of torch.Tensor
        List of target label masks for each batch in the validation set.
    temperature : float or None
        If set, logits will be divided by this temperature before loss is computed.

    Returns:
    --------
    avg_nll : float
        Average BCEWithLogitsLoss over the validation set.
    """
    logits, labels = prepare_logits_and_labels(logits, labels, temperature)
    return torch.nn.BCEWithLogitsLoss(reduction="mean")(logits, labels).item()


def compute_ece(logits: torch.Tensor, labels: torch.Tensor, temperature=None, n_bins=30):
    """
    Computes Expected Calibration Error (ECE).

    Parameters:
    -----------
    logits : list of torch.Tensor
        List of predicted probabilities (after sigmoid) for each batch.
    mask : list of torch.Tensor
        List of ground truth label masks for each batch.
    temperature : float or None
        If provided, logits will be divided by temperature before sigmoid.

    n_bins : int
        Number of bins for ECE calculation.

    Returns:
    --------
    ece : float
        Expected Calibration Error.
    """
    logits, labels = prepare_logits_and_labels(logits, labels, temperature)
    logits = torch.sigmoid(logits)

    # Bin boundaries
    bin_boundaries = torch.linspace(0, 1, steps=n_bins + 1)
    ece = torch.zeros(1).to(logits.device)

    for i in range(n_bins):
        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]

        # Get indices for the bin
        mask = (logits > lower) & (logits <= upper)
        bin_size = mask.sum()

        if bin_size > 0:
            bin_confidence = logits[mask].mean()
            bin_accuracy = labels[mask].mean()
            ece += (bin_size.float() / logits.numel()) * torch.abs(bin_confidence - bin_accuracy)

    return ece.item()
