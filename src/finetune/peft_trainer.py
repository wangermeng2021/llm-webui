

from abc import ABC, abstractmethod
class PeftTrainer((ABC)):

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def train(self):
        pass

