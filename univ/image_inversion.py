import torch

from robustness import attack_steps
from robustness.attacker import AttackerModel
from torch import Tensor
from tqdm import tqdm
from typing import Tuple


# Code taken from https://haydn.fgl.dev/posts/a-better-index-of-similarity/
class L2MomentumStep(attack_steps.AttackerStep):
    """L2 Momentum for faster convergence of inversion process"""

    def __init__(
        self,
        orig_input: Tensor,
        eps: float,
        step_size: float,
        use_grad: bool = True,
        momentum: float = 0.9,
    ):
        super().__init__(orig_input, eps, step_size, use_grad=use_grad)

        self.momentum_g = torch.zeros_like(orig_input)
        self.gamma = momentum

    def project(self, x: Tensor) -> Tensor:
        """Ensures inversion does not go outside of `self.eps` L2 ball"""

        diff = x - self.orig_input
        diff = diff.renorm(p=2, dim=0, maxnorm=self.eps)
        return torch.clamp(self.orig_input + diff, 0, 1)

    def step(self, x: Tensor, g: Tensor) -> Tensor:
        """Steps along gradient with L2 momentum"""

        g = g / g.norm(dim=(1, 2, 3), p=2, keepdim=True)
        self.momentum_g = self.momentum_g * self.gamma + g * (1.0 - self.gamma)

        return x + self.momentum_g * self.step_size


def inversion_loss(
    model: AttackerModel, inp: Tensor, targ: Tensor, vit:bool
) -> Tuple[Tensor, None]:
    """L2 distance between target representation and current inversion representation"""

    if vit:
        rep = model.forward_features(inp)
        rep = model.head(rep, pre_logits=True)
    else:
        _, rep = model(inp, with_latent=True, fake_relu=False)
    loss = torch.div(torch.norm(rep - targ, dim=1), torch.norm(targ, dim=1))
    return loss, None


def invert_images(
    model: AttackerModel,
    seed_images: Tensor,
    target_images: Tensor,
    batch_size: int = 32,
    step_size: float = 1.0 / 8.0,
    iterations: int = 2_000,
    use_best: bool = True,
    device: torch.Tensor = None,
) -> Tensor:
    """
    Representation inversion process as described in
    `If You've Trained One You've Trained Them All: Inter-Architecture Similarity Increases With Robustness`

    Default hyperparameters are exactly as used in paper.

    Parameters
    ----------
    `model` : AttackerModel
        Model to invert through, should be a robustness.attacker.AttackerModel
    `seed_images` : Tensor
        Tensor of seed images, [B, C, H, W]
    `target_images` : Tensor
        Tensor of corresponding target images [B, C, H, W]
    `batch_size` : int, optional
        Number of images to invert at once
    `step_size` : float, optional
    'learning rate' of backprop step
    `iterations` : int, optional
        Number of back prop iterations
    `use_best` : bool
        Use best inversion found rather than last

    Returns
    -------
    Tensor
        Resulting inverted images [B, C, H, W]
    """

    # L2 Momentum step
    def constraint(orig_input, eps, step_size):
        return L2MomentumStep(orig_input, eps, step_size)

    # Arguments for inversion
    kwargs = {
        "constraint":  constraint,
        "step_size":   step_size,
        "iterations":  iterations,
        "eps":         1000, # Set to large number as we are not constraining inversion
        "custom_loss": inversion_loss,
        "targeted":    True, # Minimize loss
        "use_best":    use_best,
        "do_tqdm":     False,
    }

    # Batch input
    seed_batches = seed_images.split(batch_size)
    target_batches = target_images.split(batch_size)

    # Begin inversion process
    inverted = []
    for init_imgs, targ_imgs in tqdm(
        zip(seed_batches, target_batches),
        total=len(seed_batches),
        leave=True,
        desc="Inverting",
    ):
        if device is None:
            # Get activations from target images
            (_, rep_targ), _ = model(targ_imgs.cuda(), with_latent=True)

            # Minimize distance from seed representation to target representation
            (_, _), inv = model(
                init_imgs.cuda(), rep_targ, make_adv=True, with_latent=True, **kwargs
            )
        else:
            (_, rep_targ), _ = model(targ_imgs.to(device), with_latent=True)
            (_, _), inv = model(
                init_imgs.to(device), rep_targ, make_adv=True, with_latent=True, **kwargs
            )

        inverted.append(inv.detach().cpu())
        targ_imgs.detach()
        rep_targ.detach()
        init_imgs.detach()
        del targ_imgs, rep_targ, init_imgs, inv

    inverted = torch.vstack(inverted)
    return inverted
