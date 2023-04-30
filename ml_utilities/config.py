from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import torch


@dataclass
class NameAndKwargs:
    name: str
    kwargs: Optional[Dict[str, Any]] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    entity: Optional[str]
    project_name: str
    experiment_tag: Optional[str]
    experiment_type: Optional[str]
    experiment_name: str
    experiment_dir: Optional[str]
    experiment_notes: Optional[str]
    seed: int
    gpu_id: Optional[int]
    job_name: Optional[str]
    hostname: Optional[str]


@dataclass
class ResumeTrainingConfig:
    job_dir: str
    checkpoint_idx: int


@dataclass
class TrainingStrategyConfig:
    """Configuration for DDP and training precision.
    Default configuration is for single GPU training with float32 precision."""
    enable_mixed_precision: bool = False
    precision: str = 'float32'  # 'float32', 'float16' or 'bfloat16'
    enable_autocast_gradscaler: bool = True  # passed to constructor of autocast and gradscaler
    precision_dtype: Optional[
        torch.dtype] = None  # passed to constructor of autocast and gradscaler

    def __post_init__(self):
        dtypes = {
            'float32': torch.float32,
            'bfloat16': torch.bfloat16,
            'float16': torch.float16
        }
        assert self.precision in dtypes, \
            f"precision must be one of 'float32', 'float16' or 'bfloat16' but got {self.precision}"
        self.precision_dtype = dtypes[self.precision]


@dataclass
class TrainerConfig:
    optimizer: Optional[NameAndKwargs] = None
    training_setup: Optional[str] = None
    batch_size: Optional[int] = -1
    n_steps: Optional[Union[int, float]] = -1
    n_epochs: Optional[Union[int, float]] = -1
    val_every: Union[int, float] = 1
    save_every: Union[int, float] = 0
    save_every_idxes: List[Union[int, float]] = field(default_factory=list)
    log_train_step_every: Union[int, float] = 1
    early_stopping_patience: Union[int, float] = -1
    num_workers: int = 0
    resume_training: Optional[ResumeTrainingConfig] = None
    training_strategy: Optional[TrainingStrategyConfig] = field(
        default_factory=TrainingStrategyConfig)
    lr_scheduler: Optional[NameAndKwargs] = None
    lr_scheduler_step: Optional[str] = 'step'  # 'step' or 'epoch'
    loss: str = ''
    gradient_clip_norm: Optional[float] = None
    metrics: List[Any] = field(default_factory=list)
    additional_cfg: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Config:
    experiment_data: ExperimentConfig
    trainer: TrainerConfig
    model: NameAndKwargs
    data: NameAndKwargs
    wandb: Optional[Dict[str, Any]] = None
