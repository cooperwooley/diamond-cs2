from functools import partial
from pathlib import Path
import shutil
import time
from typing import List, Optional, Tuple

from hydra.utils import instantiate
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import wandb

from agent import Agent
from data import BatchSampler, collate_segments_to_batch, Dataset, DatasetTraverser, CS2Hdf5Dataset
from utils import (
    broadcast_if_needed,
    build_ddp_wrapper,
    CommonTools,
    configure_opt,
    count_parameters,
    get_lr_sched,
    keep_agent_copies_every,
    Logs,
    move_opt_to,
    process_confusion_matrices_if_any_and_compute_classification_metrics,
    save_info_for_import_script,
    save_with_backup,
    set_seed,
    StateDictMixin,
    try_until_no_except,
    wandb_log,
)


class Trainer(StateDictMixin):
    def __init__(self, cfg: DictConfig, root_dir: Path) -> None:
        torch.backends.cuda.matmul.allow_tf32 = True
        OmegaConf.resolve(cfg)
        self._cfg = cfg
        self._rank = dist.get_rank() if dist.is_initialized() else 0
        self._world_size = dist.get_world_size() if dist.is_initialized() else 1

        set_seed(torch.seed() % 10 ** 9)

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu", self._rank)
        print(f"Starting on {self._device}")
        self._use_cuda = self._device.type == "cuda"
        if self._use_cuda:
            torch.cuda.set_device(self._rank)

        if self._rank == 0:
            try_until_no_except(
                partial(wandb.init, config=OmegaConf.to_container(cfg, resolve=True), reinit=True, resume=True, **cfg.wandb)
            )

        self._is_static_dataset = cfg.static_dataset.path is not None

        # Checkpointing
        self._path_ckpt_dir = Path("checkpoints")
        self._path_state_ckpt = self._path_ckpt_dir / "state.pt"
        self._keep_agent_copies = partial(
            keep_agent_copies_every,
            every=cfg.checkpointing.save_agent_every,
            path_ckpt_dir=self._path_ckpt_dir,
            num_to_keep=cfg.checkpointing.num_to_keep,
        )
        self._save_info_for_import_script = partial(
            save_info_for_import_script, run_name=cfg.wandb.name, path_ckpt_dir=self._path_ckpt_dir
        )

        if not cfg.common.resume and self._rank == 0:
            self._path_ckpt_dir.mkdir(exist_ok=False, parents=False)
            path_config = Path("config") / "trainer.yaml"
            path_config.parent.mkdir(exist_ok=False, parents=False)
            shutil.move(".hydra/config.yaml", path_config)
            wandb.save(str(path_config))

        # CS2 dataset setup
        assert cfg.env.train.id == "cs2", "This trainer is configured for CS2"
        assert cfg.env.path_data_low_res is not None and cfg.env.path_data_full_res is not None, \
            "Set path_data_low_res and path_data_full_res in config/env/cs2.yaml"
        assert self._is_static_dataset

        num_actions = cfg.env.num_actions
        dataset_full_res = CS2Hdf5Dataset(Path(cfg.env.path_data_full_res))

        num_workers = cfg.training.num_workers_data_loaders
        use_manager = cfg.training.cache_in_ram and (num_workers > 0)
        p = Path(cfg.static_dataset.path) if self._is_static_dataset else Path("dataset")
        self.train_dataset = Dataset(p / "train", dataset_full_res, "train_dataset", cfg.training.cache_in_ram, use_manager)
        self.test_dataset = Dataset(p / "test", dataset_full_res, "test_dataset", cache_in_ram=True)
        self.train_dataset.load_from_default_path()
        self.test_dataset.load_from_default_path()

        # Create models
        self.agent = Agent(instantiate(cfg.agent, num_actions=num_actions)).to(self._device)
        self._agent = build_ddp_wrapper(**self.agent._modules) if dist.is_initialized() else self.agent

        if cfg.initialization.path_to_ckpt is not None:
            self.agent.load(**cfg.initialization)

        # Optimizers and LR schedulers
        def build_opt(name: str) -> torch.optim.AdamW:
            return configure_opt(getattr(self.agent, name), **getattr(cfg, name).optimizer)

        def build_lr_sched(name: str) -> torch.optim.lr_scheduler.LambdaLR:
            return get_lr_sched(self.opt.get(name), getattr(cfg, name).training.lr_warmup_steps)

        model_names = ["denoiser", "upsampler"]
        self._model_names = [name for name in model_names if getattr(self.agent, name) is not None]

        self.opt = CommonTools(**{name: build_opt(name) for name in self._model_names})
        self.lr_sched = CommonTools(**{name: build_lr_sched(name) for name in self._model_names})

        # Data loaders
        make_data_loader = partial(
            DataLoader,
            dataset=self.train_dataset,
            collate_fn=collate_segments_to_batch,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0),
            pin_memory=self._use_cuda,
            pin_memory_device=str(self._device) if self._use_cuda else "",
        )

        make_batch_sampler = partial(BatchSampler, self.train_dataset, self._rank, self._world_size)

        c = cfg.denoiser.training
        seq_length = cfg.agent.denoiser.inner_model.num_steps_conditioning + 1 + c.num_autoregressive_steps
        bs = make_batch_sampler(c.batch_size, seq_length, None)
        dl_denoiser_train = make_data_loader(batch_sampler=bs)
        dl_denoiser_test = DatasetTraverser(self.test_dataset, c.batch_size, seq_length)

        if self.agent.upsampler is not None:
            c = cfg.upsampler.training
            seq_length = cfg.agent.upsampler.inner_model.num_steps_conditioning + 1 + c.num_autoregressive_steps
            bs = make_batch_sampler(c.batch_size, seq_length, None)
            dl_upsampler_train = make_data_loader(batch_sampler=bs)
            dl_upsampler_test = DatasetTraverser(self.test_dataset, c.batch_size, seq_length)
        else:
            dl_upsampler_train = dl_upsampler_test = None

        self._data_loader_train = CommonTools(dl_denoiser_train, dl_upsampler_train)
        self._data_loader_test = CommonTools(dl_denoiser_test, dl_upsampler_test)

        # Setup training
        sigma_distribution_cfg = instantiate(cfg.denoiser.sigma_distribution)
        sigma_distribution_cfg_upsampler = instantiate(cfg.upsampler.sigma_distribution) if self.agent.upsampler is not None else None
        self.agent.setup_training(sigma_distribution_cfg, sigma_distribution_cfg_upsampler)

        # Training state
        self.epoch = 0
        self.num_batch_train = CommonTools(0, 0, 0)
        self.num_batch_test = CommonTools(0, 0, 0)

        if cfg.common.resume:
            self.load_state_checkpoint()
        else:
            self.save_checkpoint()

        if self._rank == 0:
            for name in self._model_names:
                print(f"{count_parameters(getattr(self.agent, name))} parameters in {name}")
            print(self.train_dataset)
            print(self.test_dataset)

    def run(self) -> None:
        to_log = []
        num_epochs = self._cfg.training.num_final_epochs

        while self.epoch < num_epochs:
            self.epoch += 1
            start_time = time.time()

            if self._rank == 0:
                print(f"\nEpoch {self.epoch} / {num_epochs}\n")

            if self._cfg.training.should:
                to_log += self.train_agent()

            # Evaluation
            should_test = self._rank == 0 and self._cfg.evaluation.should and (self.epoch % self._cfg.evaluation.every == 0)
            if should_test:
                to_log += self.test_agent()

            # Logging
            to_log.append({"duration": (time.time() - start_time) / 3600})
            if self._rank == 0:
                wandb_log(to_log, self.epoch)
            to_log = []

            # Checkpointing
            self.save_checkpoint()

            if dist.is_initialized():
                dist.barrier()

    def train_agent(self) -> Logs:
        self.agent.train()
        self.agent.zero_grad()
        to_log = []
        for name in self._model_names:
            cfg = getattr(self._cfg, name).training
            if self.epoch > cfg.start_after_epochs:
                steps = cfg.steps_first_epoch if self.epoch == 1 else cfg.steps_per_epoch
                to_log += self.train_component(name, steps)
        return to_log

    @torch.no_grad()
    def test_agent(self) -> Logs:
        self.agent.eval()
        to_log = []
        for name in self._model_names:
            cfg = getattr(self._cfg, name).training
            if self.epoch > cfg.start_after_epochs:
                to_log += self.test_component(name)
        return to_log

    def train_component(self, name: str, steps: int) -> Logs:
        cfg = getattr(self._cfg, name).training
        model = getattr(self._agent, name)
        opt = self.opt.get(name)
        lr_sched = self.lr_sched.get(name)
        data_loader = self._data_loader_train.get(name)

        torch.cuda.empty_cache()
        model.to(self._device)
        move_opt_to(opt, self._device)

        model.train()
        opt.zero_grad()
        data_iterator = iter(data_loader) if data_loader is not None else None
        to_log = []

        num_steps = cfg.grad_acc_steps * steps

        for i in trange(num_steps, desc=f"Training {name}", disable=self._rank > 0):
            batch = next(data_iterator).to(self._device) if data_iterator is not None else None
            loss, metrics = model(batch) if batch is not None else model()
            loss.backward()

            num_batch = self.num_batch_train.get(name)
            metrics[f"num_batch_train_{name}"] = num_batch
            self.num_batch_train.set(name, num_batch + 1)

            if (i + 1) % cfg.grad_acc_steps == 0:
                if cfg.max_grad_norm is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm).item()
                    metrics["grad_norm_before_clip"] = grad_norm

                opt.step()
                opt.zero_grad()

                if lr_sched is not None:
                    metrics["lr"] = lr_sched.get_last_lr()[0]
                    lr_sched.step()

            to_log.append(metrics)

        process_confusion_matrices_if_any_and_compute_classification_metrics(to_log)
        to_log = [{f"{name}/train/{k}": v for k, v in d.items()} for d in to_log]

        model.to("cpu")
        move_opt_to(opt, "cpu")

        return to_log

    @torch.no_grad()
    def test_component(self, name: str) -> Logs:
        model = getattr(self.agent, name)
        data_loader = self._data_loader_test.get(name)
        model.eval()
        model.to(self._device)
        to_log = []
        for batch in tqdm(data_loader, desc=f"Evaluating {name}"):
            batch = batch.to(self._device)
            _, metrics = model(batch)
            num_batch = self.num_batch_test.get(name)
            metrics[f"num_batch_test_{name}"] = num_batch
            self.num_batch_test.set(name, num_batch + 1)
            to_log.append(metrics)

        process_confusion_matrices_if_any_and_compute_classification_metrics(to_log)
        to_log = [{f"{name}/test/{k}": v for k, v in d.items()} for d in to_log]
        model.to("cpu")
        return to_log

    def load_state_checkpoint(self) -> None:
        self.load_state_dict(torch.load(self._path_state_ckpt, map_location=self._device, weights_only=False))

    def save_checkpoint(self) -> None:
        if self._rank == 0:
            save_with_backup(self.state_dict(), self._path_state_ckpt)
            self.train_dataset.save_to_default_path()
            self.test_dataset.save_to_default_path()
            self._keep_agent_copies(self.agent.state_dict(), self.epoch)
            self._save_info_for_import_script(self.epoch)
