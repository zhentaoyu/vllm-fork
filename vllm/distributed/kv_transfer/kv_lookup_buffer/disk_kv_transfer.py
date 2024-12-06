import os
from typing import List, Optional, Union

import torch
import hashlib

from vllm.distributed.kv_transfer.kv_lookup_buffer.base import (
    KVLookupBufferBase)
from vllm.logger import init_logger
import time

logger = init_logger(__name__)

class DiskKVTransfer(KVLookupBufferBase):
    """
        disk kv transfer database: (tensor_key, tensor.pt) --> (file_name, file)
        1. prefill instance: hpu <--> cpu <--> disk
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
                    logger.warning(f"Can not find DiskKVTransfer work_dir_path {self.work_dir}.")
            self.init_work_dir = True

    def _get_roi_tokens(self, input_tokens: torch.Tensor, roi: torch.Tensor) -> torch.Tensor:
        # longest continuous tokens
        roi_indices = torch.arange(0, roi.size(-1), device=input_tokens.device)
        roi_tokens = input_tokens.index_select(-1, roi_indices)
        return roi_tokens

    def _encode_tensors(self, roi_tokens: torch.Tensor) -> str:

        t0 = time.perf_counter()
        combined_bytes = roi_tokens.cpu().numpy().tobytes()
        # torch.hpu.synchronize()
        logger.debug(f"2 tensors to bytes time is {time.perf_counter() - t0}")

        t0 = time.perf_counter()
        hash_object = hashlib.sha256(combined_bytes)
        hash_value = hash_object.hexdigest()
        logger.debug(f"gen hash key time is {time.perf_counter() - t0}")

        return hash_value

    def _get_tensot_key(self, input_tensor: torch.Tensor) -> str:
        tensor_key = self._encode_tensors(input_tensor) + "_" + str(self.local_rank)
        return tensor_key

    def _save_tensor(self, tensor: torch.Tensor, name: str,
                     device: Union[str, torch.device]) -> None:

        assert isinstance(tensor, torch.Tensor)
        t0 = time.perf_counter()
        tensor_device = tensor.to(device)
        logger.debug(f"tensor to cpu time is {time.perf_counter() - t0}")
        t0 = time.perf_counter()
        torch.save(tensor_device, name)
        logger.debug(f"tensor save time is {time.perf_counter() - t0}")
        logger.debug(f"local_rank: {self.local_rank}, rank: {self.rank} "\
                     f"save tensor with shape {tensor.shape} into {name}")

    def _load_tensor(self, name: str, device: Union[str, torch.device]) -> torch.Tensor:

        try:
            if not os.path.exists(name):
                logger.warning(f"cache path {name} does not exist.")
                return None
            t0 = time.perf_counter()
            tensor_cpu = torch.load(name)
            logger.debug(f"tensor load time is {time.perf_counter() - t0}")
            logger.debug(f"local_rank: {self.local_rank}, rank: {self.rank} "\
                        f"load tensor with shape {tensor_cpu.shape} from {name}")
            t0 = time.perf_counter()
            tensortt = tensor_cpu.to(device)
            logger.debug(f"tensor to hpu time is {time.perf_counter() - t0}")
            return tensortt
            # return tensor_cpu.to(device)
        except Exception as e:
            logger.error("Encountering exception in KV loading")
            logger.error("%s", e)
            return None

    def insert(self, input_tokens: torch.Tensor, roi: torch.Tensor,
               key: torch.Tensor, value: torch.Tensor,
               hidden: torch.Tensor) -> None:

        self._work_dir_init("save")

        t0 = time.perf_counter()
        roi_tokens = self._get_roi_tokens(input_tokens, roi)
        tensor_key = self._get_tensot_key(roi_tokens)
        logger.debug(f"tensor hash time is {time.perf_counter() - t0}")

        self._save_kv_to_disk(key, value, hidden, tensor_key)

    def drop_select(self, input_tokens: torch.Tensor,
                    roi: torch.Tensor,
                    load_chunk_cache: bool = False) -> List[Optional[torch.Tensor]]:

        self._work_dir_init("load")
        self.recv_device = input_tokens.device

        t0 = time.perf_counter()
        roi_tokens = self._get_roi_tokens(input_tokens, roi)
        tensor_key = self._get_tensot_key(roi_tokens)
        logger.debug(f"tensor hash time is {time.perf_counter() - t0}")

        chunk_pos_ids = None
        if not load_chunk_cache:
            key, val, hid = self._load_kv_from_disk(tensor_key)
        else:
            # load piecewise kv cache and concat them
            key, val, hid, chunk_pos_ids = self._load_chunk_kv_and_concat(input_tokens, roi)

        res = [input_tokens, roi, key, val, hid, chunk_pos_ids]
        if any(r is None for r in res):
            res = [None] * 6

        return res

    def close(self):
        pass

    def _load_kv_from_disk(self, tensor_key):
        key_path = os.path.join(self.work_dir, tensor_key + "_key.pt")
        val_path = os.path.join(self.work_dir, tensor_key + "_value.pt")
        hid_path = os.path.join(self.work_dir, tensor_key + "_hidden.pt")

        key = self._load_tensor(key_path, self.recv_device)
        val = self._load_tensor(val_path, self.recv_device)
        hid = self._load_tensor(hid_path, self.recv_device)

        return (key, val, hid)

    def _save_kv_to_disk(self, key, val, hid, tensor_key):
        key_path = os.path.join(self.work_dir, tensor_key + "_key.pt")
        val_path = os.path.join(self.work_dir, tensor_key + "_value.pt")
        hid_path = os.path.join(self.work_dir, tensor_key + "_hidden.pt")

        self._save_tensor(key, key_path, self.send_device)
        self._save_tensor(val, val_path, self.send_device)
        self._save_tensor(hid, hid_path, self.send_device)

    def _load_chunk_kv_and_concat(self, input_tokens, roi):
        # simulate chunk kv cache load
        # it's just for POC and will be replaced with kv cache transfer engine
        # modify it when knowledge content or model name change
        chunk_flags = [
            [7542, 3059, 25],    # system_prompt + rag_prefix
            [11715, 6957, 13],   # rag_0
            [311, 10515, 13],    # rag_1
            [14374, 15846, 25],  # rag_2 + prompt_prefix
        ]

        roi_tokens = self._get_roi_tokens(input_tokens, roi)
        chunk_start_pos = 0
        key, value, hidden = [], [], None
        chunk_position_ids = []
        for i in range(len(chunk_flags)):
            chunk_flag = chunk_flags[i]
            chunk_token = self._get_chunk_token(roi_tokens[chunk_start_pos:], chunk_flag)
            # no chunk matched
            # logger.debug(f"chunk_token {chunk_token}")
            if chunk_token is None:
                return (None, None, None, None)
            chunk_start_pos += len(chunk_token)
            # logger.debug(f"chunk_token length {len(chunk_token)}, chunk_start_pos {chunk_start_pos}")
            chunk_token_key = self._get_tensot_key(chunk_token)
            chunk_key, chunk_val, chunk_hid = self._load_kv_from_disk(chunk_token_key)
            chunk_pos_id = torch.arange(0, len(chunk_token), device=chunk_key.device).unsqueeze(0)
            # no found
            if any(r is None for r in [chunk_key, chunk_val, chunk_hid]):
                return (None, None, None, None)

            key.append(chunk_key)
            value.append(chunk_val)
            hidden = chunk_hid
            chunk_position_ids.append(chunk_pos_id)

        roi_key = torch.cat(key, dim=1)
        roi_value = torch.cat(value, dim=1)
        # use last chunk for hidden_states
        roi_hidden = hidden
        roi_chunk_pos_ids = torch.cat(chunk_position_ids, dim=1)

        return (roi_key, roi_value, roi_hidden, roi_chunk_pos_ids)

    def _get_chunk_token(self, roi_tokens, chunk_flag):
        roi_tokens_list = roi_tokens.to("cpu").tolist()
        roi_len = len(roi_tokens_list)
        cf_len = len(chunk_flag)
        if roi_len < cf_len:
            return None
        else:
            for i in range(roi_len):
                if i + cf_len <= roi_len:
                    cmp_token = [roi_tokens_list[i + j] for j in range(cf_len)]
                    if cmp_token == chunk_flag:
                        chunk_token = roi_tokens[:i + cf_len]
                        return chunk_token
                else:
                    return None
