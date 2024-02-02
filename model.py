import torch
import torch.optim as optim
from torch import nn
from pathlib import Path
import os
import matplotlib.pyplot as plt
from data import Data


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.fc1 = nn.Linear(64 * 5 * 5, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 64 * 5 * 5)
        return self.fc1(x)


class ImageClassifierTrainer:
    def __init__(self, model, train_set, device="cpu"):
        self.model = model
        self.train_set = train_set
        self.device = torch.device(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.model.parameters(), lr=0.001, momentum=0.9)

    def _move_model_to_device(self):
        self.model.to(self.device)

    def train(self, epochs=1, print_freq=1000):
        loss_values = []

        self._move_model_to_device()

        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(self.train_set, 1):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                if i % print_freq == 0:
                    print(
                        f'[{epoch + 1}, {i}] loss: {running_loss / print_freq:.3f}')
                    running_loss = 0.0

            # Save loss for plotting
            loss_values.append(running_loss / len(self.train_set))

            model_path = Path(os.getcwd()) / f'weights/epoch_{epoch + 1}.pth'
            torch.save(self.model.state_dict(), model_path)
            print(f'Model saved at epoch {epoch + 1}.')

        torch.save(self.model.state_dict(), Path(
            os.getcwd()) / 'weights/final.pth')

        # Plot the loss
        plt.plot(range(1, epochs + 1), loss_values, marker='o')
        plt.title('Training Loss Over Epochs')
        plt.xlabel('Epochs')
        plt.xticks(range(1, epochs + 1))
        plt.ylabel('Loss')
        plt.ylim(bottom=0)
        plt.show()

        print('Finished Training')


# Example usage:
classifier = Classifier()

data_loader = Data().train_loader

trainer = ImageClassifierTrainer(
    classifier, train_set=data_loader, device="cuda")
trainer.train(epochs=5, print_freq=500)
