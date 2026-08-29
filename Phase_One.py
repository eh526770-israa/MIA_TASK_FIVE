import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models


class Phase1:

    def __init__(self):

        # =========================================================
        # Reproducibility
        # =========================================================

        torch.manual_seed(42)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
            torch.cuda.manual_seed_all(42)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # =========================================================
        # Device
        # =========================================================

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print("Using:", self.device)

        # =========================================================
        # Transform
        # =========================================================

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor()
        ])

        # =========================================================
        # Dataset
        # =========================================================

        self.dataset = datasets.Caltech101(
            root="./data",
            download=True,
            transform=self.transform
        )

        print("Number of images:", len(self.dataset))
        print("Number of classes:", len(self.dataset.categories))
        print(self.dataset.categories)

        # =========================================================
        # Train / Validation / Test Split
        # =========================================================

        train_size = int(0.70 * len(self.dataset))
        val_size = int(0.15 * len(self.dataset))
        test_size = (
            len(self.dataset)
            - train_size
            - val_size
        )

        (
            self.train_dataset,
            self.val_dataset,
            self.test_dataset
        ) = random_split(
            self.dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )

        print("Training samples:", len(self.train_dataset))
        print("Validation samples:", len(self.val_dataset))
        print("Test samples:", len(self.test_dataset))

        # =========================================================
        # DataLoaders
        # =========================================================

        batch_size = 32

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False
        )

        print("DataLoaders are ready!")

        # =========================================================
        # Load Pretrained ResNet-34
        # =========================================================

        self.model = models.resnet34(
            weights=models.ResNet34_Weights.DEFAULT
        )

        self.model = self.model.to(self.device)

        # =========================================================
        # Freeze all layers
        # =========================================================

        for param in self.model.parameters():
            param.requires_grad = False

        # =========================================================
        # Unfreeze the last convolutional block
        # =========================================================

        for param in self.model.layer4.parameters():
            param.requires_grad = True

        # =========================================================
        # Replace Final Fully Connected Layer
        # =========================================================

        self.num_classes = len(self.dataset.categories)

        self.model.fc = nn.Linear(
            self.model.fc.in_features,
            self.num_classes
        )

        self.model.fc = self.model.fc.to(self.device)

        # The new FC layer must be trainable
        for param in self.model.fc.parameters():
            param.requires_grad = True

        # =========================================================
        # Check Trainable Parameters
        # =========================================================

        trainable_params = sum(
            p.numel()
            for p in self.model.parameters()
            if p.requires_grad
        )

        total_params = sum(
            p.numel()
            for p in self.model.parameters()
        )

        print("Total parameters:", total_params)
        print("Trainable parameters:", trainable_params)

        # =========================================================
        # Loss Function
        # =========================================================

        self.criterion = nn.CrossEntropyLoss()

        # =========================================================
        # Optimizer
        # =========================================================

        self.optimizer = optim.Adam(
            filter(
                lambda p: p.requires_grad,
                self.model.parameters()
            ),
            lr=0.0001
        )

    # =============================================================
    # Training Function
    # =============================================================

    def train_one_epoch(self):

        self.model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in self.train_loader:

            images = images.to(self.device)
            labels = labels.to(self.device)

            # Clear old gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(images)

            # Calculate loss
            loss = self.criterion(outputs, labels)

            # Backpropagation
            loss.backward()

            # Update weights
            self.optimizer.step()

            # Statistics
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(self.train_loader)
        epoch_accuracy = 100 * correct / total

        return epoch_loss, epoch_accuracy

    # =============================================================
    # Validation / Evaluation Function
    # =============================================================

    def evaluate(self, loader):

        self.model.eval()

        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():

            for images, labels in loader:

                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)

                loss = self.criterion(outputs, labels)

                running_loss += loss.item()

                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(loader)
        epoch_accuracy = 100 * correct / total

        return epoch_loss, epoch_accuracy

    # =============================================================
    # Train Model
    # =============================================================

    def train_model(self, num_epochs=5):

        for epoch in range(num_epochs):

            train_loss, train_acc = self.train_one_epoch()

            val_loss, val_acc = self.evaluate(
                self.val_loader
            )

            print(
                f"Epoch [{epoch + 1}/{num_epochs}] "
                f"| Train Loss: {train_loss:.4f} "
                f"| Train Acc: {train_acc:.2f}% "
                f"| Val Loss: {val_loss:.4f} "
                f"| Val Acc: {val_acc:.2f}%"
            )

    # =============================================================
    # Final Test Evaluation
    # =============================================================

    def test_model(self):

        test_loss, test_acc = self.evaluate(
            self.test_loader
        )

        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.2f}%")

        return test_loss, test_acc

    # =============================================================
    # Save Model
    # =============================================================

    def save_model(
        self,
        path="resnet34_caltech101_phase1.pth"
    ):

        torch.save(
            self.model.state_dict(),
            path
        )

        print("Model saved successfully!")
 

    # =============================================================
    # Run Phase 1
    # =============================================================

    def run(self, num_epochs=5):

        # Train the model
        self.train_model(num_epochs=num_epochs)

        # Evaluate on test set
        test_loss, test_acc = self.test_model()

        # Save trained model
        self.save_model()

        # Return everything needed by the next phases
        return {
            "model": self.model,
            "dataset": self.dataset,
            "train_loader": self.train_loader,
            "val_loader": self.val_loader,
            "test_loader": self.test_loader,
            "criterion": self.criterion,
            "device": self.device,
            "test_loss": test_loss,
            "test_accuracy": test_acc
        }
 
    