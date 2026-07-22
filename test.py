import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset.dataset import PhotoRestorationDataset

dataset = PhotoRestorationDataset()

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True
)

damaged, clean = next(iter(loader))

print(f"Damaged Shape : {damaged.shape}")
print(f"Clean Shape   : {clean.shape}")

damaged = damaged[0].permute(1, 2, 0).numpy()
clean = clean[0].permute(1, 2, 0).numpy()

damaged = (damaged + 1) / 2
clean = (clean + 1) / 2

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(damaged)
plt.title("Damaged Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(clean)
plt.title("Clean Image")
plt.axis("off")

plt.show()