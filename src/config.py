"""
Config class
"""

import json
import subprocess

import torch


class Config:
    """
    Configuration class
    """

    config_dict = {}

    def __init__(self, config_dict: dict):
        self.config_dict = config_dict
        self.device = self.get_device(config_dict["device"])
        self.device_map = self.get_device_map(config_dict["device_map"])
        self.torch_dtype = self.str2torchdtype(config_dict["torch_dtype"])
        self.model_name = config_dict["model_name"]

    def __str__(self):
        return str(self.to_dict())

    def __repr__(self):
        return self.__str__()

    def __eq__(self, o):
        return (
            self.config_dict == o.config_dict
            and self.device == o.device
            and self.device_map == o.device_map
            and self.torch_dtype == o.torch_dtype
        )

    def __ne__(self, o) -> bool:
        return not self.__eq__(o)

    def get_git_hash(self):
        """
        Get the git hash
        """
        # run git describe --always --dirty to get the git hash
        git_hash = (
            subprocess.check_output(["git", "describe", "--always", "--dirty"])
            .strip()
            .decode("utf-8")
        )
        return git_hash

    def get_device(self, device_str):
        """
        Get the device map
        """
        if device_str == "cpu":
            return torch.device("cpu")
        elif device_str == "cuda":
            return torch.device("cuda")
        else:
            raise ValueError(f"Unknown device map: {device_str}")

    def get_device_map(self, device_map_str):
        """
        Get the device map
        """
        if device_map_str == "default":
            return 0
        else:
            raise ValueError(f"Unknown device map: {device_map_str}")

    def str2torchdtype(self, dtype_str):
        """
        Convert a string to a torch dtype
        """
        if dtype_str == "float32":
            return torch.float32
        elif dtype_str == "float64":
            return torch.float64
        elif dtype_str == "float16":
            return torch.float16
        elif dtype_str == "bfloat16":
            return torch.bfloat16
        elif dtype_str == "int8":
            return torch.int8
        elif dtype_str == "int16":
            return torch.int16
        elif dtype_str == "int32":
            return torch.int32
        elif dtype_str == "int64":
            return torch.int64
        else:
            raise ValueError(f"Unknown dtype: {dtype_str}")

    @classmethod
    def from_file(cls, json_filename):
        """
        Create a configuration object from a json file
        """
        with open(json_filename, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        return cls(config_dict)

    @classmethod
    def from_dict(cls, config_dict):
        """
        Create a configuration object from a dictionary
        """
        return cls(config_dict)

    def to_dict(self):
        """
        Returns the configuration as a dictionary
        """
        return self.config_dict
