import torch
import torch.nn as nn


class Pix2PixLoss:
    def __init__(self, device):

        self.device = device

        self.gan_loss = nn.BCEWithLogitsLoss()

        self.l1_loss = nn.L1Loss()

        self.lambda_l1 = 100

    def discriminator_loss(
        self,
        discriminator,
        damaged,
        clean,
        fake
    ):

        real_prediction = discriminator(damaged, clean)

        fake_prediction = discriminator(
            damaged,
            fake.detach()
        )

        real_labels = torch.ones_like(
            real_prediction,
            device=self.device
        )

        fake_labels = torch.zeros_like(
            fake_prediction,
            device=self.device
        )

        real_loss = self.gan_loss(
            real_prediction,
            real_labels
        )

        fake_loss = self.gan_loss(
            fake_prediction,
            fake_labels
        )

        return (real_loss + fake_loss) / 2

    def generator_loss(
        self,
        discriminator,
        damaged,
        clean,
        fake
    ):

        prediction = discriminator(
            damaged,
            fake
        )

        real_labels = torch.ones_like(
            prediction,
            device=self.device
        )

        adversarial_loss = self.gan_loss(
            prediction,
            real_labels
        )

        reconstruction_loss = self.l1_loss(
            fake,
            clean
        )

        total_loss = (
            adversarial_loss +
            self.lambda_l1 * reconstruction_loss
        )

        return (
            total_loss,
            adversarial_loss,
            reconstruction_loss
        )