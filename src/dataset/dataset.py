from abc import ABC, abstractmethod


class Dataset(ABC):
    @abstractmethod
    def get_dataset(self):
        pass
