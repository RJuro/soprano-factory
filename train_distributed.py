"""
Distributed training script for Soprano TTS.
Supports multi-GPU training with PyTorch DDP.

Usage:
    Single GPU:  python train_distributed.py --save-dir outputs/model
    Multi-GPU:   torchrun --nproc_per_node=4 train_distributed.py --save-dir outputs/model
"""
import argparse
import os
import pathlib
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from dataset import AudioDataset


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="./coral_danish_dataset", type=pathlib.Path)
    parser.add_argument("--save-dir", required=True, type=pathlib.Path)
    parser.add_argument("--max-steps", default=10000, type=int)
    parser.add_argument("--batch-size", default=48, type=int, help="Batch size per GPU")
    parser.add_argument("--lr", default=5e-4, type=float)
    parser.add_argument("--val-freq", default=250, type=int)
    parser.add_argument("--val-steps", default=5, type=int, help="Number of validation batches to average over")
    parser.add_argument("--save-freq", default=1000, type=int)
    parser.add_argument("--save-best", action="store_true", help="Save best checkpoint based on validation loss")
    parser.add_argument("--log-file", type=pathlib.Path, help="Path to CSV file for logging metrics")
    return parser.parse_args()


def setup_distributed():
    """Initialize distributed training if available."""
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size, True
    return 0, 0, 1, False


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank):
    return rank == 0


class Trainer:
    def __init__(self, args, rank, local_rank, world_size, distributed):
        self.args = args
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.distributed = distributed
        self.device = f"cuda:{local_rank}"

        # Hyperparameters
        self.batch_size = args.batch_size
        self.seq_len = 1024
        self.max_lr = args.lr
        self.min_lr = 0.1 * self.max_lr
        self.warmup_ratio = 0.1
        self.cooldown_ratio = 0.1
        self.max_steps = args.max_steps
        self.text_factor = 0.01
        self.betas = (0.9, 0.95)
        self.weight_decay = 0.1
        self.val_steps = args.val_steps

        # Best checkpoint tracking
        self.best_val_loss = float('inf')

        # CSV logging
        self.log_file = args.log_file
        if self.log_file and is_main_process(rank):
            with open(self.log_file, 'w') as f:
                f.write("step,train_audio_loss,train_text_loss,train_acc,val_audio_loss,val_text_loss,val_acc,lr,grad_norm,tokens_per_sec\n")

        # LR schedule
        self.warmup_steps = int(self.max_steps * self.warmup_ratio)
        self.cooldown_steps = int(self.max_steps * self.cooldown_ratio)

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained('ekwek/Soprano-80M')
        self.model = AutoModelForCausalLM.from_pretrained('ekwek/Soprano-80M')
        self.model.to(torch.bfloat16).to(self.device)

        if distributed:
            self.model = DDP(self.model, device_ids=[local_rank])

        self.model.train()

        # Optimizer
        model_params = self.model.module.parameters() if distributed else self.model.parameters()
        self.optimizer = torch.optim.AdamW(
            model_params, self.max_lr,
            betas=self.betas, weight_decay=self.weight_decay, fused=True
        )

        # Dataloaders
        self._setup_dataloaders()

        if is_main_process(rank):
            print(f"Initialized training:")
            print(f"  World size: {world_size}")
            print(f"  Batch size per GPU: {self.batch_size}")
            print(f"  Effective batch size: {self.batch_size * world_size}")
            print(f"  Max steps: {self.max_steps}")

    def _setup_dataloaders(self):
        train_dataset = AudioDataset(f'{self.args.input_dir}/train.json')
        val_dataset = AudioDataset(f'{self.args.input_dir}/val.json')

        train_sampler = DistributedSampler(train_dataset, shuffle=True) if self.distributed else None
        val_sampler = DistributedSampler(val_dataset, shuffle=False) if self.distributed else None

        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size * 16,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True,
            collate_fn=self._collate_pack,
        )
        self.val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.batch_size * 16,
            shuffle=False,
            sampler=val_sampler,
            num_workers=1,
            pin_memory=True,
            collate_fn=self._collate_pack,
        )
        self.train_iter = iter(self.train_dataloader)

    def _collate_pack(self, texts):
        tokens_batch = self.tokenizer(texts, padding=False, truncation=False)
        batch = []
        cur_sample, cur_size = [], 0
        for i in range(len(texts)):
            tokens = torch.tensor(tokens_batch['input_ids'][i][:-1], dtype=torch.long)
            cur_size += tokens.size(0)
            cur_sample.append(tokens)
            if cur_size >= self.seq_len + 1:
                batch.append(torch.cat(cur_sample)[:self.seq_len + 1])
                cur_sample, cur_size = [], 0
                if len(batch) == self.batch_size:
                    break
        if cur_sample and not batch:
            batch.append(torch.cat(cur_sample + [torch.zeros(self.seq_len, dtype=torch.long)])[:self.seq_len + 1])
        if len(batch) < self.batch_size:
            pad = batch[-1] if batch else torch.zeros(self.seq_len + 1, dtype=torch.long)
            while len(batch) < self.batch_size:
                batch.append(pad)
        batch = torch.stack(batch)
        return batch[:, :-1], batch[:, 1:]

    def get_lr(self, step):
        if step < self.warmup_steps:
            return self.max_lr * (step + 1) / self.warmup_steps
        if step < self.max_steps - self.cooldown_steps:
            return self.max_lr
        return self.min_lr + (self.max_lr - self.min_lr) * ((self.max_steps - step) / self.cooldown_steps)

    def compute_loss(self, logits, y):
        pred = logits.view(-1, logits.size(-1))
        labels = y.reshape(-1)
        loss = torch.nn.functional.cross_entropy(pred, labels, reduction='none')
        audio_mask = torch.logical_and(y >= 3, y <= 8003).view(-1)
        audio_loss = loss[audio_mask].mean()
        text_loss = loss[~audio_mask].mean()
        acc = (logits.argmax(dim=-1) == y).view(-1)[audio_mask].float().mean()
        return audio_loss, text_loss, acc

    @torch.no_grad()
    def evaluate(self, step=None):
        self.model.eval()
        val_audio_loss = torch.tensor(0.0, device=self.device)
        val_text_loss = torch.tensor(0.0, device=self.device)
        val_acc = torch.tensor(0.0, device=self.device)

        val_iter = iter(self.val_dataloader)
        num_steps = min(self.val_steps, len(self.val_dataloader))

        for _ in range(num_steps):
            try:
                x, y = next(val_iter)
            except StopIteration:
                break
            x, y = x.to(self.device), y.to(self.device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = self.model(x).logits if not self.distributed else self.model.module(x).logits
                audio_loss, text_loss, acc = self.compute_loss(logits, y)
            val_audio_loss += audio_loss
            val_text_loss += text_loss
            val_acc += acc

        # Average over steps
        val_audio_loss /= num_steps
        val_text_loss /= num_steps
        val_acc /= num_steps

        if self.distributed:
            dist.all_reduce(val_audio_loss, op=dist.ReduceOp.AVG)
            dist.all_reduce(val_text_loss, op=dist.ReduceOp.AVG)
            dist.all_reduce(val_acc, op=dist.ReduceOp.AVG)

        val_audio_loss_val = val_audio_loss.item()
        val_text_loss_val = val_text_loss.item()
        val_acc_val = val_acc.item()

        if is_main_process(self.rank):
            print(f"val text loss: {val_text_loss_val:.4f} | val audio loss: {val_audio_loss_val:.4f} | val acc: {val_acc_val:.4f}")

            # Save best checkpoint
            total_val_loss = val_audio_loss_val + self.text_factor * val_text_loss_val
            if self.args.save_best and total_val_loss < self.best_val_loss:
                self.best_val_loss = total_val_loss
                best_path = f"{self.args.save_dir}_best"
                print(f"New best validation loss: {total_val_loss:.4f} - saving to {best_path}")
                model_to_save = self.model.module if self.distributed else self.model
                model_to_save.save_pretrained(best_path)
                self.tokenizer.save_pretrained(best_path)

        self.model.train()
        return val_audio_loss_val, val_text_loss_val, val_acc_val

    def save_checkpoint(self, step):
        if is_main_process(self.rank):
            save_path = f"{self.args.save_dir}_step{step}"
            print(f"Saving checkpoint to {save_path}")
            model_to_save = self.model.module if self.distributed else self.model
            model_to_save.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)

    def log_metrics(self, step, train_audio_loss, train_text_loss, train_acc,
                    val_audio_loss, val_text_loss, val_acc, lr, grad_norm, tokens_per_sec):
        if self.log_file and is_main_process(self.rank):
            with open(self.log_file, 'a') as f:
                val_audio = f"{val_audio_loss:.4f}" if val_audio_loss is not None else ""
                val_text = f"{val_text_loss:.4f}" if val_text_loss is not None else ""
                val_a = f"{val_acc:.4f}" if val_acc is not None else ""
                f.write(f"{step},{train_audio_loss:.4f},{train_text_loss:.4f},{train_acc:.4f},"
                        f"{val_audio},{val_text},{val_a},{lr:.2e},{grad_norm:.3f},{tokens_per_sec:.0f}\n")

    def train(self):
        torch.set_float32_matmul_precision('high')

        pbar = tqdm(range(self.max_steps), disable=not is_main_process(self.rank), ncols=200)
        for step in pbar:
            start = time.time()

            # Validation
            val_audio_loss, val_text_loss, val_acc = None, None, None
            if self.args.val_freq > 0 and (step % self.args.val_freq == 0 or step == self.max_steps - 1):
                val_audio_loss, val_text_loss, val_acc = self.evaluate(step)

            # Get batch
            try:
                x, y = next(self.train_iter)
            except StopIteration:
                if self.distributed:
                    self.train_dataloader.sampler.set_epoch(step)
                self.train_iter = iter(self.train_dataloader)
                x, y = next(self.train_iter)

            x, y = x.to(self.device), y.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = self.model(x).logits
                audio_loss, text_loss, acc = self.compute_loss(logits, y)

            total_loss = audio_loss + self.text_factor * text_loss
            total_loss.backward()

            norm = torch.nn.utils.clip_grad_norm_(
                self.model.module.parameters() if self.distributed else self.model.parameters(),
                1.0
            )

            # Update LR and step
            lr = self.get_lr(step)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.optimizer.step()

            # Logging
            torch.cuda.synchronize()
            dt = (time.time() - start) * 1000
            tokens_per_sec = (self.batch_size * self.seq_len * self.world_size) / (time.time() - start)

            train_audio_loss = audio_loss.item()
            train_text_loss = text_loss.item()
            train_acc = acc.item()

            pbar.set_description(
                f"loss: {train_audio_loss:.3f} | text: {train_text_loss:.3f} | "
                f"acc: {train_acc:.4f} | lr: {lr:.2e} | norm: {norm:.3f} | "
                f"{dt:.0f}ms | {tokens_per_sec:.0f} t/s"
            )

            # Log to CSV
            self.log_metrics(step, train_audio_loss, train_text_loss, train_acc,
                           val_audio_loss, val_text_loss, val_acc, lr, norm, tokens_per_sec)

            # Save checkpoint
            if self.args.save_freq > 0 and (step + 1) % self.args.save_freq == 0:
                self.save_checkpoint(step + 1)

        # Final save
        if is_main_process(self.rank):
            print(f"Training complete. Saving final model to {self.args.save_dir}")
            model_to_save = self.model.module if self.distributed else self.model
            model_to_save.save_pretrained(self.args.save_dir)
            self.tokenizer.save_pretrained(self.args.save_dir)


def main():
    args = get_args()
    rank, local_rank, world_size, distributed = setup_distributed()

    if is_main_process(rank):
        os.makedirs(args.save_dir, exist_ok=True)
        print(f"Save directory: {args.save_dir}")

    # Set seeds
    seed = 1337 + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    trainer = Trainer(args, rank, local_rank, world_size, distributed)
    trainer.train()

    cleanup_distributed()


if __name__ == '__main__':
    main()
