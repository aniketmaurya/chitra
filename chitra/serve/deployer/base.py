from abc import ABC, abstractmethod


class Deploy(ABC):
    @abstractmethod
    def upload_model(self, target_path: str):
        raise NotImplementedError

    @abstractmethod
    def dockerize(self):
        raise NotImplementedError
