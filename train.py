import os

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

from config import (
    DEVICE,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    BETA1,
    BETA2,
    CHECKPOINT_DIR
)

from dataset.dataset import PhotoRestorationDataset
from models.generator import Generator
from models.discriminator import Discriminator
from losses.loss import Pix2PixLoss


def train():

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    dataset = PhotoRestorationDataset()

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    generator = Generator().to(DEVICE)

    discriminator = Discriminator().to(DEVICE)

    optimizer_g = Adam(
        generator.parameters(),
        lr=LEARNING_RATE,
        betas=(BETA1, BETA2)
    )

    optimizer_d = Adam(
        discriminator.parameters(),
        lr=LEARNING_RATE,
        betas=(BETA1, BETA2)
    )

    criterion = Pix2PixLoss(DEVICE)

    for epoch in range(NUM_EPOCHS):

        generator.train()

        discriminator.train()

        total_g = 0

        total_d = 0

        for damaged, clean in dataloader:

            damaged = damaged.to(DEVICE)

            clean = clean.to(DEVICE)

            fake = generator(damaged)

            optimizer_d.zero_grad()

            d_loss = criterion.discriminator_loss(
                discriminator,
                damaged,
                clean,
                fake
            )

            d_loss.backward()

            optimizer_d.step()

            optimizer_g.zero_grad()

            fake = generator(damaged)

            g_loss, adv_loss, l1_loss = criterion.generator_loss(
                discriminator,
                damaged,
                clean,
                fake
            )

            g_loss.backward()

            optimizer_g.step()

            total_g += g_loss.item()

            total_d += d_loss.item()

        avg_g = total_g / len(dataloader)

        avg_d = total_d / len(dataloader)

        print(
            f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
            f"G Loss: {avg_g:.4f} "
            f"D Loss: {avg_d:.4f}"
        )

        torch.save(
            generator.state_dict(),
            os.path.join(
                CHECKPOINT_DIR,
                "generator_latest.pth"
            )
        )

        torch.save(
            discriminator.state_dict(),
            os.path.join(
                CHECKPOINT_DIR,
                "discriminator_latest.pth"
            )
        )

    print("Training Completed")


if __name__ == "__main__":
    train()