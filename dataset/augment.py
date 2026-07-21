import os
import cv2
import random
import numpy as np


# Add Gaussian noise to the image.
def add_gaussian_noise(image, mean=0, sigma=20):
    noise = np.random.normal(mean, sigma, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    noisy = np.clip(noisy, 0, 255)
    return noisy.astype(np.uint8)


# Apply horizontal motion blur.
def add_motion_blur(image, kernel_size=9):
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[(kernel_size - 1) // 2, :] = np.ones(kernel_size)
    kernel /= kernel_size
    return cv2.filter2D(image, -1, kernel)


# Draw random scratches on the image.
def add_scratches(image, num_scratches=15):
    img = image.copy()
    h, w = img.shape[:2]

    for _ in range(num_scratches):
        x1 = random.randint(0, w)
        y1 = random.randint(0, h)

        length = random.randint(30, 120)
        angle = random.uniform(0, np.pi)

        x2 = int(x1 + length * np.cos(angle))
        y2 = int(y1 + length * np.sin(angle))

        color = (
            random.randint(180, 255),
            random.randint(180, 255),
            random.randint(180, 255)
        )

        thickness = random.randint(1, 2)
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)

    return img


# Add random dust particles.
def add_dust(image, num_spots=200):
    img = image.copy()
    h, w = img.shape[:2]

    for _ in range(num_spots):
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)
        radius = random.randint(1, 3)
        color = random.randint(180, 255)
        cv2.circle(img, (x, y), radius, (color, color, color), -1)

    return img


# Simulate faded colors.
def fade_image(image, factor=0.7):
    faded = image.astype(np.float32) * factor
    faded = np.clip(faded, 0, 255)
    return faded.astype(np.uint8)


# Add JPEG compression artifacts.
def jpeg_artifacts(image, quality=25):
    _, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return cv2.imdecode(encoded, 1)


# Apply a random combination of damages.
def random_damage(image):
    img = image.copy()

    if random.random() < 0.8:
        img = add_gaussian_noise(img, sigma=random.randint(10, 25))

    if random.random() < 0.6:
        img = add_motion_blur(img, kernel_size=random.choice([5, 7, 9]))

    if random.random() < 0.7:
        img = add_scratches(img, random.randint(8, 20))

    if random.random() < 0.5:
        img = add_dust(img, random.randint(80, 250))

    if random.random() < 0.5:
        img = fade_image(img, random.uniform(0.55, 0.85))

    if random.random() < 0.5:
        img = jpeg_artifacts(img, random.randint(15, 35))

    return img


# Generate damaged images from the clean dataset.
def generate_dataset(clean_folder, damaged_folder):
    os.makedirs(damaged_folder, exist_ok=True)

    for filename in os.listdir(clean_folder):
        image = cv2.imread(os.path.join(clean_folder, filename))

        if image is None:
            continue

        damaged = random_damage(image)

        cv2.imwrite(
            os.path.join(damaged_folder, filename),
            damaged
        )

        print(f"Saved {filename}")


# Run dataset generation.
if __name__ == "__main__":
    generate_dataset("dataset/clean", "dataset/damaged")