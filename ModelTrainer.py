import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from utils import check_dir


class ModelTrainer:
    def __init__(self, pytorch_model, device=0):
        self.device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
        self.pytorch_model = pytorch_model.to(self.device)
        self.optimizer, self.criteria, self.epochs = self.set_hyperparameters()
        self.train_loss, self.val_loss = None, None
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)

    def save_model(self, model_save_path):
        check_dir(model_save_path)
        torch.save(self.pytorch_model.state_dict(), model_save_path)
        print("Finished Training, Model Saved to:", model_save_path)

    def train_model(self, train_loader, val_loader):
        self.train_loss = []
        self.val_loss = []
        self.pytorch_model.train()
        for epoch in range(self.epochs):
            running_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.pytorch_model(images)
                loss = self.criteria(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            print(f"Epoch {epoch + 1}, Loss: {running_loss/len(train_loader)}")
            self.train_loss.append(running_loss/len(train_loader))
            self.scheduler.step()

            if epoch % 5 == 4:
                accuracy, loss = self.evaluate_model(val_loader)
                self.val_loss.append(loss)
                self.pytorch_model.train()
        self.pytorch_model.eval()

    def evaluate_model(self, dataloader):
        self.pytorch_model.eval()
        correct = 0
        total = 0
        loss = 0.0
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.pytorch_model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                loss += self.criteria(outputs, labels).item()
        accuracy = 100 * correct / total
        loss = loss / len(dataloader)
        print(f"Accuracy: {accuracy:.2f}%, Loss: {loss:.4f}")
        return accuracy, loss

    def plot_loss_curve(self, loss_curve_path=None):
        time_steps_sparse = list(range(5, len(self.train_loss) + 1, 5))
        plt.plot(self.train_loss, label='Train loss')
        plt.scatter(time_steps_sparse, self.val_loss, label='Val loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        if loss_curve_path:
            check_dir(loss_curve_path)
            plt.savefig(loss_curve_path)
        plt.show()

    def set_hyperparameters(self):
        SGD = False
        if SGD:
            lr = 0.001
            momentum = 0.9
            weight_decay = 0.0001
            nesterov = True
            # optimizer = torch.optim.SGD(self.pytorch_model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
            optimizer = torch.optim.SGD(self.pytorch_model.parameters(), lr=lr, momentum=momentum)
            print_str = f"Optimizer: SGD, lr={lr}, momentum={momentum}, weight_decay={weight_decay}, nesterov={nesterov}"
        else:
            lr = 0.0001
            weight_decay = 0.0001
            optimizer = torch.optim.Adam(self.pytorch_model.parameters(), lr=lr, weight_decay=weight_decay)
            print_str = f"Optimizer: Adam, lr={lr}, weight_decay={weight_decay}"
        print(print_str)
        criteria = nn.CrossEntropyLoss()
        epochs = 30
        return optimizer, criteria, epochs
