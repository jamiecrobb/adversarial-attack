import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from pathlib import Path
import os


class Data:
    def __init__(self):
        self.data_path = Path(os.getcwd()) / "data"
        self.train_loader, self.test_loader = self.get_mnist_loaders(16)

    def get_mnist_loaders(self, batch_size: int = 16) -> (DataLoader, DataLoader):
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        train_dataset = MNIST(root=self.data_path, train=True,
                              download=True, transform=transform)
        test_dataset = MNIST(root=self.data_path, train=False,
                             download=True, transform=transform)

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False)

        return (train_loader, test_loader)

    def get_train_batch(self):
        return next(iter(self.train_loader))

    def display_n_images(self, batch, num_to_display: int = 8):
        images, labels = batch
        _, axes = plt.subplots(1, num_to_display, figsize=(12, 4))
        for i in range(num_to_display):
            image = transforms.ToPILImage()(images[i])
            axes[i].imshow(image, cmap='gray')
            axes[i].axis('off')
            axes[i].set_title(f"L: {labels[i].item()}")
        plt.show()
