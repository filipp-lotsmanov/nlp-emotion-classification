# trainer.py
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import json


class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, val_loss, epoch):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            print(f"  No improvement: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0

        return self.early_stop


class WarmupScheduler:
    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.current_step = 0

    def step(self):
        self.current_step += 1
        lr = self._get_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    def _get_lr(self):
        step = max(self.current_step, 1)
        return (self.d_model**-0.5) * min(step**-0.5, step * (self.warmup_steps**-1.5))


class TransformerTrainer:
    def __init__(self, model, train_loader, valid_loader, config, device="cuda"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device
        self.config = config

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.get("learning_rate", 1e-4),
            betas=(0.9, 0.98),
            eps=1e-9,
            weight_decay=config.get("weight_decay", 0.0001),
        )

        self.criterion = nn.CrossEntropyLoss(
            ignore_index=0, label_smoothing=config.get("label_smoothing", 0.1)
        )

        if config.get("use_warmup", True):
            self.scheduler = WarmupScheduler(
                self.optimizer,
                d_model=config.get("d_model", 512),
                warmup_steps=config.get("warmup_steps", 4000),
            )
            self.use_warmup = True
        else:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=3
            )
            self.use_warmup = False

        self.early_stopping = EarlyStopping(
            patience=config.get("early_stop_patience", 10),
            min_delta=config.get("early_stop_min_delta", 0.0001),
        )

        self.accumulation_steps = config.get("gradient_accumulation_steps", 1)

        self.use_amp = config.get("use_amp", False) and "cuda" in str(device)
        if self.use_amp:
            self.scaler = GradScaler(device="cuda")
            print("  Using mixed precision (FP16)")

        self.writer = None  # Initialize as None, can be set externally

        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.history = {"train_loss": [], "val_loss": [], "learning_rates": []}

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        self.optimizer.zero_grad()

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch + 1}")

        for batch_idx, (src, tgt) in enumerate(pbar):
            src = src.to(self.device)
            tgt = tgt.to(self.device)

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            if self.use_amp:
                with autocast(device_type="cuda"):
                    output = self.model(src, tgt_input)
                    loss = self.criterion(
                        output.reshape(-1, output.size(-1)), tgt_output.reshape(-1)
                    )

                loss = loss / self.accumulation_steps
                self.scaler.scale(loss).backward()

                if (batch_idx + 1) % self.accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                    if self.use_warmup:
                        current_lr = self.scheduler.step()

                    self.optimizer.zero_grad()
            else:
                output = self.model(src, tgt_input)
                loss = self.criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))

                loss = loss / self.accumulation_steps
                loss.backward()

                if (batch_idx + 1) % self.accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                    if self.use_warmup:
                        current_lr = self.scheduler.step()

                    self.optimizer.zero_grad()

            total_loss += loss.item() * self.accumulation_steps
            self.global_step += 1

            pbar.set_postfix({"loss": f"{loss.item() * self.accumulation_steps:.4f}"})

            if self.writer and self.global_step % 50 == 0:
                self.writer.add_scalar("train/loss", loss.item(), self.global_step)
                self.writer.add_scalar(
                    "train/lr", self.optimizer.param_groups[0]["lr"], self.global_step
                )

        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for src, tgt in tqdm(self.valid_loader, desc="Validating"):
                src = src.to(self.device)
                tgt = tgt.to(self.device)

                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]

                output = self.model(src, tgt_input)
                loss = self.criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))

                total_loss += loss.item()

        return total_loss / len(self.valid_loader)

    def train(self, num_epochs, save_path="checkpoints/best_model.pt"):
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        for epoch in range(num_epochs):
            self.epoch = epoch

            train_loss = self.train_epoch()
            val_loss = self.validate()

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            print(f"\nEpoch {epoch + 1}: Train={train_loss:.4f}, Val={val_loss:.4f}")

            if not self.use_warmup:
                self.scheduler.step(val_loss)

            if self.writer:
                self.writer.add_scalar("epoch/train_loss", train_loss, epoch)
                self.writer.add_scalar("epoch/val_loss", val_loss, epoch)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_loss": val_loss,
                        "config": self.config,
                        "history": self.history,
                    },
                    save_path,
                )
                print(f"  *** Best model saved: {val_loss:.4f} ***")

            if (epoch + 1) % 5 == 0:
                checkpoint_path = Path(save_path).parent / f"checkpoint_epoch_{epoch + 1}.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                    },
                    checkpoint_path,
                )

            if self.early_stopping(val_loss, epoch):
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

        if self.writer:
            self.writer.close()

        with open(Path(save_path).parent / "history.json", "w") as f:
            json.dump(self.history, f, indent=2)
