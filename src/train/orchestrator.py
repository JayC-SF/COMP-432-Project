import src.variables as v
from pathlib import Path
from src.train.history import TrainingHistory
import torch
import copy
from tqdm.auto import tqdm
from sklearn.metrics import classification_report, confusion_matrix, f1_score


class Orchestrator:
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        train_loader,
        val_loader,
        device,
        patience,
        save_path,
        scheduler,
        max_epochs,
        classes
    ):

        self.save_path = save_path
        self.th = TrainingHistory(self.save_path, model, optimizer, device, scheduler, recover=self.save_path.exists())
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.patience = patience
        self.max_epochs = max_epochs
        self.classes = classes

    def train(self):
        print(f"Running with device:{self.device}")
        self.th.model = self.th.model.to(self.device)
        continue_training = True
        while continue_training:
            self.th.epoch += 1
            print(f"---- Starting Epoch {self.th.epoch} ----")

            # save lrs in history
            current_lr = self.th.optimizer.param_groups[0]['lr']
            self.th.lrs.append(current_lr)

            self.train_step()
            val_loss = self.validate_step()
            # Step the scheduler if it exists
            if self.th.scheduler:
                old_lr = current_lr
                # ReduceLROnPlateau needs val_loss, others don't
                if isinstance(self.th.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.th.scheduler.step(val_loss)
                else:
                    self.th.scheduler.step()

                new_lr = self.th.optimizer.param_groups[0]['lr']

                if new_lr < old_lr:
                    print(f"📉 Learning rate reduced to {new_lr:.2e}")

            self.th.save_checkpoint()
            continue_training = self.early_stopping_check(val_loss)

        print(f"Completed training at epoch {self.th.epoch}")

    def train_step(self):
        self.th.model.train()  # Set model to training mode
        running_loss = 0
        running_corrects = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.th.epoch} [Train]", unit="batch", leave=False)

        for inputs, labels in pbar:
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

        print(f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4%}")

    def validate_step(self):
        # 1. Set model to evaluation mode
        self.th.model.eval()

        running_loss = 0
        running_corrects = 0

        # 2. Turn off the gradient engine (saves memory/time)
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {self.th.epoch} [Validate]", unit="batch", leave=False)
            for inputs, labels in pbar:
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

        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4%}")
        return val_loss  # Return this so your early stopping can check it

    def early_stopping_check(self, val_loss):
        if val_loss < self.th.best_val_loss:
            print(f"🌟 New Best Model! Val Loss decreased from {self.th.best_val_loss:.4f} to {val_loss:.4f}")
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

    def test(self, test_loader):
        # 1. Set model to evaluation mode
        self.th.model.eval()

        running_loss = 0
        running_corrects = 0
        all_preds = []
        all_labels = []
        # 2. Turn off the gradient engine (saves memory/time)
        with torch.no_grad():
            pbar = tqdm(test_loader, desc=f"Epoch {self.th.epoch} [Test]", unit="batch", leave=False)
            for inputs, labels in pbar:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # 3. Forward pass ONLY
                outputs = self.th.model(inputs)
                loss = self.criterion(outputs, labels)

                # 4. Record statistics
                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                running_corrects += (preds == labels).sum().item()

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # 5. Calculate final averages for the epoch
        results = {}
        results['test_loss'] = running_loss / len(test_loader.dataset)
        results['test_acc'] = running_corrects / len(test_loader.dataset)

        print(f"Test Loss: {results['test_loss']:.4f} | Test Acc: {results['test_acc']:.4%}")

        results['classification_report'] = classification_report(all_labels, all_preds, target_names=self.classes)
        results['confusion_matrix'] = confusion_matrix(all_labels, all_preds)

        return results
