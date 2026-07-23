import os
import torch


def save_checkpoint(
    generator,
    discriminator,
    optimizer_g,
    optimizer_d,
    epoch,
    path
):
    checkpoint = {
        "epoch": epoch,
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "optimizer_g": optimizer_g.state_dict(),
        "optimizer_d": optimizer_d.state_dict()
    }

    torch.save(checkpoint, path)

    print(f"Checkpoint saved to {path}")


def load_checkpoint(
    path,
    generator,
    discriminator,
    optimizer_g=None,
    optimizer_d=None
):

    if not os.path.exists(path):
        print("Checkpoint not found.")
        return 0

    checkpoint = torch.load(path)

    generator.load_state_dict(checkpoint["generator"])
    discriminator.load_state_dict(checkpoint["discriminator"])

    if optimizer_g is not None:
        optimizer_g.load_state_dict(checkpoint["optimizer_g"])

    if optimizer_d is not None:
        optimizer_d.load_state_dict(checkpoint["optimizer_d"])

    print(f"Checkpoint loaded from {path}")

    return checkpoint["epoch"]