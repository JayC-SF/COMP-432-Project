import torch
import os


class TrainingHistory:
    def __init__(self, save_path, model, optimizer, device, scheduler=None, recover=False):
        self.save_path = save_path
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        if recover:
            self.recover(device)
        else:
            self.train_loss = []
            self.val_loss = []
            self.train_acc = []
            self.val_acc = []
            self.lrs = []
            self.best_val_loss = float('inf')
            self.early_stopping_counter = 0
            self.epoch = 0
            self.best_model_weights = None

    def save_checkpoint(self):
        """Saves everything needed to resume training if the system crashes."""
        # Exclude the heavy model/optimizer objects and the static 'best' weights
        exclude = ['model', 'optimizer', 'best_model_weights', 'scheduler']
        save_data = {k: v for k, v in vars(self).items() if k not in exclude}

        # Inject the live states
        save_data['model_state'] = self.model.state_dict()
        save_data['optimizer_state'] = self.optimizer.state_dict()
        if self.scheduler is not None:
            save_data['scheduler'] = self.scheduler.state_dict()

        self.save_path.mkdir(parents=True, exist_ok=True)
        torch.save(save_data, self.save_path / 'latest_history.pt')
        print(f"Saved checkpoint data under {self.save_path / 'latest_history.pt'}")

    def save_best(self):
        """Saves only the best weights. Call this only when val_loss improves."""
        if self.best_model_weights is not None:
            torch.save(self.best_model_weights, self.save_path / 'best_model.pt')
            print(f"Saved best_model_weights under {self.save_path / 'best_model.pt'}")

    def recover(self, device):
        """Recovers the state saved on disk.

        Raises:
            TypeError: If a scheduler was found on disk but not in the instance
            TypeError: If a scheduler was not found on disk but found in the instance
        """
        print("🚀 Recovering state from disk...")
        checkpoint = torch.load(self.save_path / 'latest_history.pt', weights_only=False, map_location=device)

        # Restore the basic stats
        for key, value in checkpoint.items():
            if key not in ['model_state', 'optimizer_state', 'scheduler']:
                setattr(self, key, value)

        # Restore the weights into the live objects
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.to(device)
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        if 'scheduler' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        elif 'scheduler' in checkpoint and self.scheduler is None:
            raise TypeError("Scheduler was found in checkpoint but not found in instance configuration.")
        elif 'scheduler' not in checkpoint and self.scheduler is not None:
            raise TypeError("Scheduler was not found in checkpoint but found in instance configuration.")
        # Check if a best model exists to keep the memory consistent
        try:
            self.best_model_weights = torch.load(self.save_path / 'best_model.pt', weights_only=False, map_location=device)
        except FileNotFoundError:
            self.best_model_weights = None
