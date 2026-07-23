from pathlib import Path
import matplotlib.pyplot as plt
import torch
def denormalize(image):
    image = image.detach().cpu()
    image = image * 0.5 + 0.5
    image = image.clamp(0, 1)
    return image
def save_image_grid(
    damaged,
    generated,
    clean,
    epoch,
    save_dir,
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    damaged = denormalize(damaged[0]).permute(1, 2, 0)
    generated = denormalize(generated[0]).permute(1, 2, 0)
    clean = denormalize(clean[0]).permute(1, 2, 0)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    images = [damaged, generated, clean]
    titles = ["Damaged", "Generated", "Clean"]
    for ax, image, title in zip(axes, images, titles):
        ax.imshow(image)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(
        save_dir / f"epoch_{epoch:03d}.png",
        dpi=200
    )
    plt.close()