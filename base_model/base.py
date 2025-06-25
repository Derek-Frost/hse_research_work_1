from abc import ABC, abstractmethod


class BaseModel(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_model_id(self):
        raise NotImplementedError

    @abstractmethod
    def get_model_name(self):
        raise NotImplementedError

    @abstractmethod
    def get_param_file(self, param_folder_path=""):
        raise NotImplementedError
