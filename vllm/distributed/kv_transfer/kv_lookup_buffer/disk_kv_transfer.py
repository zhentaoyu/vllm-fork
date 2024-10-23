import os
from typing import List, Optional, Union

import torch
import hashlib

from vllm.distributed.kv_transfer.kv_lookup_buffer.base import (
    KVLookupBufferBase)
from vllm.logger import init_logger

logger = init_logger(__name__)

class DiskKVTransfer(KVLookupBufferBase):
    """
        disk kv transfer database: (tensor_key, tensor.pt) --> (file_name, file)
        1. prefill instance: hpu --> cpu --> disk
        2. decode instance: disk --> cpu --> hpu
    """

    def __init__(self, local_rank: int, root_dir: str = "") -> None:

        self.root_dir = root_dir if root_dir is not None else ""
        self.work_dir = os.path.join(self.root_dir, "disagg_prefill_kv")

        self.init_work_dir = False
        self.local_rank = local_rank
        self.rank = torch.distributed.get_rank()
        self.send_device = "cpu"
        self.recv_device = None

    def _work_dir_init(self, load_or_save: str) -> None:
        if not self.init_work_dir:
            if not os.path.exists(self.work_dir):
                if load_or_save == "save":
                    os.makedirs(self.work_dir)
                    logger.debug(f"DiskKVTransfer work_dir_path {self.work_dir} has been created.")
                else:
                    assert load_or_save == "load"
                    raise RuntimeError(f"Can not find DiskKVTransfer work_dir_path {self.work_dir}.")
            self.init_work_dir = True

    def _encode_tensors(self, tensor1: torch.Tensor, tensor2: torch.Tensor) -> str:

        tensor1_clone = tensor1.clone()
        tensor2_clone = tensor2.clone()
        tensor1_bytes = tensor1_clone.cpu().numpy().tobytes()
        tensor2_bytes = tensor2_clone.cpu().numpy().tobytes()

        combined_bytes = tensor1_bytes + tensor2_bytes
        hash_object = hashlib.sha256(combined_bytes)
        hash_value = hash_object.hexdigest()

        return hash_value

    def _save_tensor(self, tensor: torch.Tensor, name: str,
                     device: Union[str, torch.device]) -> None:

        assert isinstance(tensor, torch.Tensor)
        tensor_device = tensor.to(device)
        torch.save(tensor_device, name)
        logger.debug(f"local_rank: {self.local_rank}, rank: {self.rank} "\
                     f"save tensor with shape {tensor.shape} into {name}")

    def _load_tensor(self, name: str, device: Union[str, torch.device]) -> torch.Tensor:

        tensor_cpu = torch.load(name, weights_only=True)
        logger.debug(f"local_rank: {self.local_rank}, rank: {self.rank} "\
                     f"load tensor with shape {tensor_cpu.shape} from {name}")
        return tensor_cpu.to(device)

    def insert(self, input_tokens: torch.Tensor, roi: torch.Tensor,
               key: torch.Tensor, value: torch.Tensor,
               hidden: torch.Tensor) -> None:

        self._work_dir_init("save")

        tensor_key = self._encode_tensors(input_tokens, roi) + "_" + str(self.local_rank)
        key_path = os.path.join(self.work_dir, tensor_key + "_key.pt")
        val_path = os.path.join(self.work_dir, tensor_key + "_value.pt")
        hid_path = os.path.join(self.work_dir, tensor_key + "_hidden.pt")

        self._save_tensor(key, key_path, self.send_device)
        self._save_tensor(value, val_path, self.send_device)
        self._save_tensor(hidden, hid_path, self.send_device)

    def drop_select(self, input_tokens: torch.Tensor,
                    roi: torch.Tensor) -> List[Optional[torch.Tensor]]:

        self._work_dir_init("load")

        tensor_key = self._encode_tensors(input_tokens, roi) + "_" + str(self.local_rank)
        key_path = os.path.join(self.work_dir, tensor_key + "_key.pt")
        val_path = os.path.join(self.work_dir, tensor_key + "_value.pt")
        hid_path = os.path.join(self.work_dir, tensor_key + "_hidden.pt")

        self.recv_device = input_tokens.device
        key = self._load_tensor(key_path, self.recv_device)
        val = self._load_tensor(val_path, self.recv_device)
        hid = self._load_tensor(hid_path, self.recv_device)
        assert key is not None
        res = [input_tokens, roi, key, val, hid]

        return res

    def close(self):
        pass
