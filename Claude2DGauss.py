"""
2D Gaussian Splatting from an Image — CPU-Compatible (No CUDA Required)
========================================================================
Fits a set of 2D Gaussian splats to a target image using gradient descent.

Requirements:
    pip install torch torchvision pillow matplotlib numpy

Example Usage:
    python Claude2DGauss.py --folder folder_name --n_gaussians 1000 --steps 2000 --max_index 100 --start_index 50
"""

import argparse
import math

import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
import cv2
from PIL import Image


# ---------------------------------------------------------------------------
# Device selection — prefer MPS (Apple Silicon) > CPU; never requires CUDA
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        print("Using CUDA (Nvidia GPU)")
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        print("Using MPS (Apple Silicon GPU)")
        return torch.device("mps")
    print("Using CPU")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Loading all images in the folder of choice (if it exists)
# ---------------------------------------------------------------------------

def load_folder(folder_name: str, folder_type:str, max_index: int, starting_index: int = 0):
    """Returns the desired folder path, and the list of filenames in that folder (inside a defined slice)"""
    folder_path = os.path.join('Output_Data', folder_name, folder_type)
    filenames = os.listdir(folder_path)[starting_index: max_index]
    return folder_path, filenames

# ---------------------------------------------------------------------------
# Gaussian Splat model
# ---------------------------------------------------------------------------

class GaussianSplats2D(nn.Module):
    """
    Learnable set of 2D Gaussian splats.

    Each splat has:
      - means      : (N, 2)   — (x, y) center in pixel coordinates
      - log_scales : (N, 2)   — log of (sx, sy); exponentiated during render
      - rotations  : (N,)     — rotation angle in radians
      - colors     : (N, 3)   — RGB, passed through sigmoid → [0, 1]
      - opacities  : (N,)     — passed through sigmoid → [0, 1]
    """

    def __init__(self, n_gaussians: int, H: int, W: int, device: torch.device):
        super().__init__()
        self.H = H
        self.W = W
        self.device = device

        # Spread means uniformly over the image canvas
        self.means = nn.Parameter(
            torch.rand(n_gaussians, 2, device=device) * torch.tensor([W, H], device=device)
        )
        # Start with small-to-medium splats (log scale ≈ log(5..20 px))
        self.log_scales = nn.Parameter(
            torch.rand(n_gaussians, 2, device=device) * 2.0 + 1.5  # ~ exp(1.5..3.5)
        )
        self.rotations = nn.Parameter(
            torch.rand(n_gaussians, device=device) * 2 * math.pi
        )
        self.colors = nn.Parameter(torch.rand(n_gaussians, 3, device=device))
        # Start opacities near 0.5 (logit(0.5) = 0)
        self.opacities = nn.Parameter(torch.zeros(n_gaussians, device=device))

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self, batch_size: int = 64) -> torch.Tensor:
        """
        Rasterize all Gaussians onto an (H, W, 3) image.

        Gaussians are processed in batches to cap peak memory usage,
        which is important on CPU where RAM is the bottleneck.

        Returns:
            image : (H, W, 3) float32 tensor in [0, 1]
        """
        H, W = self.H, self.W
        device = self.device

        # Pixel coordinate grid — shape (H, W, 2)  [x=col, y=row]
        ys, xs = torch.meshgrid(
            torch.arange(H, dtype=torch.float32, device=device),
            torch.arange(W, dtype=torch.float32, device=device),
            indexing="ij",
        )
        pixels = torch.stack([xs, ys], dim=-1)          # (H, W, 2)

        # Accumulated colour buffer
        image = torch.zeros(H, W, 3, device=device)

        N = self.means.shape[0]
        scales = torch.exp(self.log_scales)              # (N, 2)
        cos_r = torch.cos(self.rotations)                # (N,)
        sin_r = torch.sin(self.rotations)                # (N,)

        # Covariance:  Σ = R @ diag(s²) @ Rᵀ  (closed-form elements)
        sx2 = scales[:, 0] ** 2
        sy2 = scales[:, 1] ** 2
        S00 = cos_r ** 2 * sx2 + sin_r ** 2 * sy2
        S01 = cos_r * sin_r * (sx2 - sy2)
        S11 = sin_r ** 2 * sx2 + cos_r ** 2 * sy2

        # Inverse of 2×2 symmetric matrix
        det = S00 * S11 - S01 ** 2 + 1e-8
        I00 =  S11 / det
        I01 = -S01 / det
        I11 =  S00 / det

        colors   = torch.sigmoid(self.colors)            # (N, 3)
        alphas   = torch.sigmoid(self.opacities)         # (N,)

        # Process in batches to limit peak memory
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            b = end - start

            # Offset from each Gaussian centre to every pixel
            # d : (b, H, W, 2)
            d = pixels.unsqueeze(0) - self.means[start:end].view(b, 1, 1, 2)
            dx = d[..., 0]    # (b, H, W)
            dy = d[..., 1]

            # Mahalanobis distance squared
            maha = (
                I00[start:end].view(b, 1, 1) * dx ** 2
                + 2.0 * I01[start:end].view(b, 1, 1) * dx * dy
                + I11[start:end].view(b, 1, 1) * dy ** 2
            )                                            # (b, H, W)

            # Gaussian weight × opacity
            weight = alphas[start:end].view(b, 1, 1) * torch.exp(-0.5 * maha)

            # Accumulate weighted colours  (additive blending)
            image = image + (weight.unsqueeze(-1) * colors[start:end].view(b, 1, 1, 3)).sum(0)

        return image.clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def load_image(path: str, max_size: int = 256) -> torch.Tensor:
    """Load an image as a float32 tensor in [0, 1] with shape (H, W, 3)."""
    img = Image.open(path).convert("RGB")

    # Optionally downscale so training is fast even on CPU
    # w, h = img.size
    # scale = min(max_size / max(h, w), 1.0)
    # if scale < 1.0:
    #     new_w, new_h = int(w * scale), int(h * scale)
    #     img = img.resize((new_w, new_h), Image.LANCZOS) # type: ignore
    #     print(f"Resized image to {new_w}×{new_h}")

    return torch.from_numpy(np.array(img) / 255.0).float()


def train(
    image_path: str,
    n_gaussians: int = 1000,
    n_steps: int = 2000,
    lr: float = 5e-3,
    batch_size: int = 64,
    max_image_size: int = 256,
    save_path: str = "result.png",
    show_every: int = 500,
):
    device = get_device()

    # ---- Load target image ------------------------------------------------
    target = load_image(image_path, max_size=max_image_size).to(device)
    H, W, _ = target.shape
    print(f"Target image: {H}×{W}, fitting {n_gaussians} Gaussians for {n_steps} steps")

    # ---- Model & optimiser ------------------------------------------------
    model = GaussianSplats2D(n_gaussians, H, W, device)

    # Different learning rates per parameter group for faster convergence
    optimizer = optim.Adam([
        {"params": model.means,       "lr": lr * 10},   # positions move fast
        {"params": model.log_scales,  "lr": lr},
        {"params": model.rotations,   "lr": lr},
        {"params": model.colors,      "lr": lr},
        {"params": model.opacities,   "lr": lr},
    ])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps)

    # ---- Training loop ----------------------------------------------------
    losses = []
    for step in range(1, n_steps + 1):
        optimizer.zero_grad()

        rendered = model.render(batch_size=batch_size)        # (H, W, 3)

        # L2 loss + small L1 regularisation on opacity (keeps splats sparse)
        loss = nn.functional.mse_loss(rendered, target)
        loss = loss + 1e-4 * torch.mean(torch.sigmoid(model.opacities))

        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())

        if step % 100 == 0 or step == 1:
            print(f"  Step {step:5d}/{n_steps}  loss={loss.item():.5f}")

        if step % show_every == 0 or step == n_steps:
            save_image(tensor=rendered.permute(2, 0, 1), fp=save_path)
            #_show_progress(target, rendered, losses, step, save_path)

    print(f"\nDone! Final render saved to '{save_path}'")
    return model, losses


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def _show_progress(target, rendered, losses, step, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].imshow(target.detach().cpu().numpy())
    axes[0].set_title("Target"); axes[0].axis("off")

    axes[1].imshow(rendered.detach().cpu().numpy())
    axes[1].set_title(f"Rendered (step {step})"); axes[1].axis("off")

    axes[2].plot(losses)
    axes[2].set_title("Loss"); axes[2].set_xlabel("Step"); axes[2].set_ylabel("MSE")
    axes[2].set_yscale("log")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Fit 2D Gaussian splats to an image (CPU-friendly)")
    p.add_argument("--folder",        type=str,   required=True,  help="Path to residuals folder")
    p.add_argument("--type",          type=str,   required=True,  choices=['residuals', 'frames'] ,help="residuals or frames?")
    p.add_argument("--n_gaussians",   type=int,   default=1000,   help="Number of Gaussian splats")
    p.add_argument("--steps",         type=int,   default=2000,   help="Optimisation steps")
    p.add_argument("--max_index",     type=int,   required=True,  help="The last image you want to compute")
    p.add_argument("--start_index",   type=int,   default=0,      help="The image you want to start computations on")
    p.add_argument("--lr",            type=float, default=5e-3,   help="Base learning rate")
    p.add_argument("--max_size",      type=int,   default=256,    help="Max image dimension (resize if larger)")
    p.add_argument("--batch_size",    type=int,   default=64,     help="Gaussians per render batch (lower = less RAM)")
    p.add_argument("--output",        type=str,   default="result.png", help="Output image path")
    p.add_argument("--show_every",    type=int,   default=500,    help="Save progress image every N steps")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if (args.max_index < args.start_index) or args.max_index < 0 or args.start_index < 0:
        print("Error in defined indexes!")
        exit()
    
    folder_path, image_list = load_folder(args.folder, args.type, args.max_index)
    splat_path = os.path.join('Splats', args.folder, args.type, str(args.n_gaussians))
    if not os.path.exists(splat_path):
        print(f'Creating Splats folder at {splat_path}...')
        os.makedirs(splat_path)
    else:
        print(f'Adding to folder {splat_path}...')

    for filename in image_list:
        full_path = os.path.join(folder_path, filename)
        splat_name = f'{args.folder}_splat_{args.type}_{args.n_gaussians}{filename[-8:-4]}'
        if os.path.exists(os.path.join(splat_path, f'{splat_name}.png')):
            print(f'Splat at {full_path} already exists, skipping...')
            continue
        else:
            print(f'Computing 2D Gaussian for {filename}...')
        train(
            image_path    = full_path,
            n_gaussians   = args.n_gaussians,
            n_steps       = args.steps,
            lr            = args.lr,
            batch_size    = args.batch_size,
            max_image_size= args.max_size,
            save_path     = os.path.join(splat_path, f'{splat_name}.png'),
            show_every    = args.show_every,
        )
    print("Calculations complete!")
