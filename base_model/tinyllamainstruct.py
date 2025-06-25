import os
from .base import BaseModel


class LocalTinyLlamaModel(BaseModel):
    def __init__(self, model_name="TinyLlama-1.1B", model_folder="./models"):
        """
        Класс для работы с локальной моделью TinyLlama, хранящейся в формате safetensors.
        model_name: Название модели.
        model_folder: Папка, где хранятся файлы модели.
        """
        self.model_name = model_name
        self.model_folder = model_folder
        self.param_file = f"{model_name}.safetensors"
        self.dec_param_file_n = "llama_decomposed_params.pt"

    def get_model_id(self):
        """Возвращает полный путь к папке модели."""
        return self.model_folder

    def get_model_name(self):
        return self.model_name

    def get_param_file(self):
        """Возвращает полный путь к файлу параметров модели."""
        return os.path.join(self.model_folder, self.param_file)

    def get_dec_param_file(self):
        """Возвращает полный путь к файлу с декомпозированными параметрами (.pt)."""
        return os.path.join(self.model_folder, self.dec_param_file_n)

    def model_exists(self):
        """Проверяет, существует ли файл модели в указанной директории."""
        return os.path.exists(self.get_param_file())
