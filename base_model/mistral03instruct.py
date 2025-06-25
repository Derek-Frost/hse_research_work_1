import os

from .base import BaseModel


class MistralV03Instruct7B(BaseModel):
    def __init__(self):
        self.model_id = "mistralai/Mistral-7B-Instruct-v0.3"
        self.dec_param_file_n = "mistral_decomposed_params.pt"

    def get_model_id(self):
        return self.model_id

    def get_model_name(self):
        return self.model_id.split("/")[1]

    def get_param_file(self, param_folder_path=""):
        return os.path.join(param_folder_path, self.dec_param_file_n)
