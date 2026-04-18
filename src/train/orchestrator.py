import secrets
import src.variables as v
from pathlib import Path
from src.train.history import TrainingHistory
import torch
import copy


class TrainOrchestrator:
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        train_loader,
        val_loader,
        device,
        max_epochs,
        patience,
        save_path,
    ):

        self.save_path = save_path
        self.th = TrainingHistory(self.save_path, model, optimizer, recover=self.save_path.exists())
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.max_epochs = max_epochs
        self.patience = patience

    def train(self):
        self.th.model = self.th.model.to(self.device)
        continue_training = True
        while continue_training:
            self.th.epoch += 1
            print(f"---- Starting Epoch {self.th.epoch} ----")
            self.train_step()
            val_loss = self.validate_step()
            self.th.save_checkpoint()
            continue_training = self.early_stopping_check(val_loss)

        print(f"Completed training at epoch {self.th.epoch}")

    def train_step(self):
        self.th.model.train()  # Set model to training mode
        running_loss = 0
        running_corrects = 0

        for inputs, labels in self.train_loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.th.optimizer.zero_grad()
            outputs = self.th.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.th.optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += (preds == labels).sum().item()

        epoch_loss = running_loss / len(self.train_loader.dataset)
        epoch_acc = running_corrects / len(self.train_loader.dataset)

        # --- Store in your History object ---
        self.th.train_loss.append(epoch_loss)
        self.th.train_acc.append(epoch_acc)

        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

    def validate_step(self):
        # 1. Set model to evaluation mode
        self.th.model.eval()

        running_loss = 0
        running_corrects = 0

        # 2. Turn off the gradient engine (saves memory/time)
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # 3. Forward pass ONLY
                outputs = self.th.model(inputs)
                loss = self.criterion(outputs, labels)

                # 4. Record statistics
                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                running_corrects += (preds == labels).sum().item()

        # 5. Calculate final averages for the epoch
        val_loss = running_loss / len(self.val_loader.dataset)
        val_acc = running_corrects / len(self.val_loader.dataset)

        # 6. Save to history
        self.th.val_loss.append(val_loss)
        self.th.val_acc.append(val_acc)

        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        return val_loss  # Return this so your early stopping can check it

    def early_stopping_check(self, val_loss):
        if val_loss < self.th.best_val_loss:
            print(f"🌟 New Best Model! Loss decreased from {self.th.best_val_loss:.4f} to {val_loss:.4f}")
            self.th.best_val_loss = val_loss
            self.th.early_stopping_counter = 0

            self.th.best_model_weights = copy.deepcopy(self.th.model.state_dict())
            self.th.save_best()
        else:
            self.th.early_stopping_counter += 1
            print(f"⚠️ No improvement. Early Stopping Counter: {self.th.early_stopping_counter}/{self.patience}")

        # Stop training if we run out of patience or hit max_epochs
        if self.th.early_stopping_counter >= self.patience:
            print("🛑 Early stopping triggered.")
            return False

        if self.th.epoch >= self.max_epochs:
            print("🏁 Max epochs reached.")
            return False
        return True
