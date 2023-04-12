from abc import ABC, abstractmethod

from torch.utils import data
from torchmetrics import MetricCollection


class DatasetGeneratorInterface(ABC):

    def generate_dataset(self) -> None:
        pass
    
    @property
    @abstractmethod
    def dataset_generated(self) -> bool:
        pass

    @property
    @abstractmethod
    def train_split(self) -> data.Dataset:
        pass

    @property
    @abstractmethod
    def val_split(self) -> data.Dataset:
        pass

    @property
    def train_metrics(self) -> MetricCollection:
        return None

    @property
    def val_metrics(self) -> MetricCollection:
        return None