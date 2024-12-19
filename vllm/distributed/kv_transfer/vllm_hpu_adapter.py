"""vLLM distributed KV cache transfer API.
These APIs are used in `vllm/worker/hpu_model_runner.py`.

Currently supporting TP. The TP between prefill and decode instance needs to be
the same.

Workflow (disaggregated prefill)
- In prefill instance
    - After prefill, vLLM `insert` its KV caches into a lookup buffer.
    - The prefill instance will also open up a thread that listens to
      `drop_select` request.
- In decode instance
    - vLLM first runs `drop_select` to send input tokens and a mask on input
      tokens (we call it roi, region of interest) to prefill instance
    - The prefill instance then respond to `drop_select` request by
        - Finding a match in current lookup buffer.
        - Clone and send the matched item out
        - Delete the matched item in the lookup buffer to free up GPU memory.
    - The decode vLLM then store the KV cache into paged memory.
"""
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

from copy import deepcopy
import itertools

import torch
from torch.distributed import Backend

from vllm_hpu_extension import cache_ops as ops

if TYPE_CHECKING:
    from vllm.worker.hpu_model_runner import (ModelInputForHPUWithSamplingMetadata, HpuModelAdapter)


import vllm.envs as envs
from vllm.distributed.kv_transfer.kv_lookup_buffer.base import (
    KVLookupBufferBase)
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors

import time
logger = init_logger(__name__)

# check VLLM_DISTRIBUTERD_KV_ROLE and set corresponding flags
assert envs.VLLM_DISTRIBUTED_KV_ROLE in [None, "producer", "consumer", "both"],\
    "VLLM_DISTRIBUTERD_KV_ROLE can only be producer, consumer or both."
IS_DISTRIBUTED_KV_INSTANCE: bool = (envs.VLLM_DISTRIBUTED_KV_ROLE
                                    in ["producer", "consumer", "both"])
IS_KV_PRODUCER: bool = (envs.VLLM_DISTRIBUTED_KV_ROLE in ["producer", "both"])
IS_KV_CONSUMER: bool = (envs.VLLM_DISTRIBUTED_KV_ROLE in ["consumer", "both"])

# When the current instance is both KV producer and KV consumer,
# it is likely connected to a KV storage service on CPU/disk
# so the communication backend needs to be "gloo" for that case.

DISTRIBUTED_BACKEND: str = "gloo" if (IS_KV_PRODUCER
                                      and IS_KV_CONSUMER) else "hccl"
# corresponding device
DISTRIBUTED_DEVICE: str = "cpu" if (IS_KV_PRODUCER
                                    and IS_KV_CONSUMER) else "hpu"


class KV_transfer_agent:
    """
    A class designated for distributed KV transfer

    Target use cases:
        1. Disaggregated prefill
        2. Remote KV cache storage
    """

    def __init__(
        self,
        group_ranks: List[List[int]],
        local_rank: int,
        torch_distributed_backend: Union[str, Backend] = DISTRIBUTED_BACKEND,
        # FIXME(Kuntai): remove this hardcoding
        lookup_buffer_size: int = int(1e10)):

        self.lookup_buffer_size = lookup_buffer_size

        self.send_buffer: Optional[KVLookupBufferBase] = None
        self.recv_buffer: Optional[KVLookupBufferBase] = None

        assert envs.VLLM_KV_TRANSFER_DRIVER in ["simple_buffer", "disk_kv_transfer"], \
            "VLLM_KV_TRANSFER_DRIVER can only be simple_buffer or disk_kv_transfer."
        self.kv_transfer_driver = envs.VLLM_KV_TRANSFER_DRIVER
        logger.debug(f"kv_transfer_driver is {self.kv_transfer_driver}")

        self.load_chunk_cache = (envs.VLLM_KV_TRANSFER_CHUNK_CACHE and IS_KV_PRODUCER)
        logger.debug(f"load_chunk_cache is {self.load_chunk_cache}")

        # TODO other block size
        self.block_size = 128
        # TODO remove
        self.original_input_tokens_t = None
        self.original_slot_mapping = None

        if self.kv_transfer_driver == "simple_buffer":
            import vllm.distributed.kv_transfer.kv_lookup_buffer.simple_buffer as sklb
            from vllm.distributed.kv_transfer.kv_pipe.torch_distributed_pipe import (
                TorchDistributedPipe)

            SimpleKVLookupBuffer = sklb.SimpleKVLookupBuffer

            # In disaggregated prefill, the prefill vLLM only uses send pipe
            # and the decode vLLM only uses recv pipe
            # In remote KV cache store, vLLM will use both send pipe and recv pipe
            # So we build both send pipe and recv pipe for simplicity.
            if IS_KV_PRODUCER:

                self.send_pipe = TorchDistributedPipe(
                    group_ranks,
                    local_rank,
                    DISTRIBUTED_BACKEND,
                )
                self.send_signal_pipe = TorchDistributedPipe(
                    group_ranks,
                    local_rank,
                    "gloo",
                )
                self.recv_pipe = TorchDistributedPipe(
                    group_ranks,
                    local_rank,
                    DISTRIBUTED_BACKEND,
                )
                self.recv_signal_pipe = TorchDistributedPipe(
                    group_ranks,
                    local_rank,
                    "gloo",
                )
                self.send_buffer = SimpleKVLookupBuffer(self.send_signal_pipe,
                                                        self.send_pipe,
                                                        self.lookup_buffer_size)
                self.recv_buffer = SimpleKVLookupBuffer(self.recv_signal_pipe,
                                                        self.recv_pipe,
                                                        self.lookup_buffer_size)
                self.tensor_device = DISTRIBUTED_DEVICE
            else:

                # the current vLLM instance is KV consumer, so it needs to connect
                # its recv pipe to the send pipe of KV producder

                self.recv_pipe = TorchDistributedPipe(
                    group_ranks,
                    local_rank,
                    DISTRIBUTED_BACKEND,
                )
                self.recv_signal_pipe = TorchDistributedPipe(
                    group_ranks,
                    local_rank,
                    "gloo",
                )
                self.send_pipe = TorchDistributedPipe(
                    group_ranks,
                    local_rank,
                    DISTRIBUTED_BACKEND,
                )
                self.send_signal_pipe = TorchDistributedPipe(
                    group_ranks,
                    local_rank,
                    "gloo",
                )
                self.send_buffer = SimpleKVLookupBuffer(self.send_signal_pipe,
                                                        self.send_pipe,
                                                        self.lookup_buffer_size)
                self.recv_buffer = SimpleKVLookupBuffer(self.recv_signal_pipe,
                                                        self.recv_pipe,
                                                        self.lookup_buffer_size)
                self.tensor_device = DISTRIBUTED_DEVICE

        elif self.kv_transfer_driver == "disk_kv_transfer":
            from vllm.distributed.kv_transfer.kv_lookup_buffer.disk_kv_transfer import DiskKVTransfer

            self.send_buffer = DiskKVTransfer(local_rank)
            self.recv_buffer = DiskKVTransfer(local_rank)

        else:
            raise ValueError("Invalid kv_transfer_driver.")

    def send_kv_caches_and_hidden_states(
        self,
        model_executable: "HpuModelAdapter",
        model_input: "ModelInputForHPUWithSamplingMetadata",
        kv_caches: List[torch.Tensor],
        hidden_or_intermediate_states: Union[torch.Tensor,
                                             IntermediateTensors],
    ) -> None:

        torch_model = model_executable.model
        input_tokens_tensor = model_input.input_tokens
        seq_lens = model_input.seq_lens
        # query_lens = model_input.query_lens
        slot_mapping = model_input.attn_metadata.slot_mapping
        # TODO remove
        if self.original_input_tokens_t is not None:
            input_tokens_tensor = self.original_input_tokens_t
            slot_mapping = self.original_slot_mapping
        try:
            start_layer = torch_model.model.start_layer
            end_layer = torch_model.model.end_layer
        except:
            # no pipeline parallelism
            start_layer = 0
            end_layer = len(torch_model.model.layers)

        # query_lens contains new KV caches that are added to vLLM.
        # so we will send them to decode instance
        # FIXME(Kuntai): This assume that all requests are prefill.
        logger.debug(f"seq_lens tensor in send {input_tokens_tensor.size(-1)} {seq_lens}")
        # ignore padded request
        send_seq_lens = seq_lens[:model_input.real_batch_size]
        for idx, slen in enumerate(send_seq_lens):
            start_pos = 0
            # # TODO prompt attention with context (query_lens)
            end_pos = start_pos + slen
            # select tokens
            # hpu use [bs, seqlen + padding] in prefill stage
            # only store kv cache and related states without padding for general hash key
            select_index = torch.arange(start_pos, end_pos, device=input_tokens_tensor.device)
            current_tokens = input_tokens_tensor[idx].index_select(-1, select_index)

            keys, values = [], []

            for layer_id in range(start_layer, end_layer):
                kv_cache = kv_caches[layer_id - start_layer]

                _, _, num_heads, head_size = kv_cache[0].shape

                key_cache = kv_cache[0].reshape(-1, num_heads, head_size)
                value_cache = kv_cache[1].reshape(-1, num_heads, head_size)

                current_slot_mapping = slot_mapping[idx][start_pos:end_pos]

                keys.append(key_cache[current_slot_mapping].unsqueeze(0))
                values.append(value_cache[current_slot_mapping].unsqueeze(0))

            keys = torch.cat(keys, dim=0)
            values = torch.cat(values, dim=0)
            # FIXME (maybe) trim_logits in hpu
            hidden_send = hidden_or_intermediate_states[idx].reshape(1, -1)

            if self.send_buffer is not None:
                logger.debug(f"kv role {envs.VLLM_DISTRIBUTED_KV_ROLE} starts to send kv")
                # hccl can not send bool, so use int type instead
                roi = torch.ones_like(current_tokens, dtype=torch.int32)
                self.send_buffer.insert(current_tokens, roi, keys, values, hidden_send)

        logger.debug("[rank%d]: KV send DONE.", torch.distributed.get_rank())

    def destroy(self) -> None:
        if self.send_buffer is not None:
            self.send_buffer.close()
        if self.recv_buffer is not None:
            self.recv_buffer.close()

    # both prefill instance and decode instance may receive (part) kv cache
    # in their first-token stages
    def recv_kv_caches_and_hidden_states(
        self,
        model_executable: "HpuModelAdapter",
        model_input: "ModelInputForHPUWithSamplingMetadata",
        kv_caches: List[torch.Tensor]
    ) -> Tuple[Union[torch.Tensor, IntermediateTensors], bool,
               "ModelInputForHPUWithSamplingMetadata"]:

        # When this flag is set to False, it means that at least for one
        # request its corresponding KV cache or hidden state is missing.
        # In this case we need to do prefilling to recompute missing KV cache
        # and hidden states.
        bypass_model_exec = True

        torch_model = model_executable.model
        input_tokens_tensor = model_input.input_tokens
        seq_lens = model_input.seq_lens
        slot_mapping = model_input.attn_metadata.slot_mapping
        # TODO remove it for unified API
        rag_start_pos = [-1] * input_tokens_tensor.size(0)
        rag_end_pos = [-1] * input_tokens_tensor.size(0)
        # split in prefill instance
        if hasattr(self, "rag_end_pos") and IS_KV_PRODUCER:
            rag_start_pos = self.rag_start_pos
            rag_end_pos = self.rag_end_pos

        hidden_or_intermediate_states_for_one_req = []

        input_tokens_list = []
        num_computed_tokens_list = []
        start_pos_list = []

        try:
            start_layer = torch_model.model.start_layer
            end_layer = torch_model.model.end_layer
        except:
            # no pipeline parallelism
            start_layer = 0
            end_layer = len(torch_model.model.layers)

        # enumerate different requests
        # FIXME(Kuntai): This impl assumes that all requests are prefill.
        logger.debug(f"seq_lens tensor {input_tokens_tensor.size(-1)} {seq_lens}")

        # deal with padded bs
        real_bs = model_input.real_batch_size
        # block_indices = model_input.attn_metadata.block_indices
        # block_offsets = model_input.attn_metadata.block_offsets
        # logger.debug(f"input tokens shape is {input_tokens_tensor.shape}, "
        #              f"block_indices is {block_indices}, "
        #              f"block_offsets is {block_offsets}")

        for idx, slen in enumerate(seq_lens):

            # hpu use [bs, seqlen + padding] in prefill stage
            start_pos = 0
            end_pos = start_pos + slen
            select_index = torch.arange(start_pos, end_pos, device=input_tokens_tensor.device)
            current_tokens = input_tokens_tensor[idx].index_select(-1, select_index)

            num_tokens = slen

            # collecting data for rebuilding the input
            input_tokens_list.append(current_tokens)
            start_pos_list.append(start_pos)

            if self.recv_buffer is None:
                bypass_model_exec = False
                break

            logger.debug(f"start_pos {start_pos}, slen {slen}")
            logger.debug(f"kv role {envs.VLLM_DISTRIBUTED_KV_ROLE} starts to receive kv")
            # 1. no need to care about padded requests kv cache
            # 2. make a padded hidden_states for sampling if all real requests have
            #    fetched whole kv caches successfully
            if idx >= real_bs:
                if bypass_model_exec:
                    hidden = torch.zeros_like(hidden_or_intermediate_states_for_one_req[0])
                    hidden_or_intermediate_states_for_one_req.append(hidden)
                # remain one token for batch hidden_states generation
                num_computed_tokens_list.append(end_pos - 1)
            else:
                roi_len = slen if rag_end_pos[idx] == -1 else (rag_end_pos[idx] + 1)
                 # hccl can not send bool, so use int type instead
                roi = torch.ones(roi_len, dtype=torch.int32, device=current_tokens.device)
                ret = self.recv_buffer.drop_select(current_tokens,
                                                   roi,
                                                   load_chunk_cache=self.load_chunk_cache)
                if ret[0] is None:
                    # didn't find any match.
                    bypass_model_exec = False
                    num_computed_tokens_list.append(0)
                    continue

                roi: torch.Tensor = ret[1]
                keys: torch.Tensor = ret[2]
                values: torch.Tensor = ret[3]
                hidden: torch.Tensor = ret[4]

                num_computed_tokens = roi.shape[-1]
                num_computed_tokens_list.append(num_computed_tokens)

                # check if both KV cache and the hidden states are received
                # If not, need to redo the forwarding to compute missing states
                hit_cache = True if (keys is not None and values is not None) else False
                # redo all prompt forward if any part of key or value is missing
                if hit_cache:
                    if keys.size() != values.size():
                        hit_cache = False

                if not all([(num_computed_tokens == num_tokens), hidden is not None
                            ]):
                    bypass_model_exec = False

                # skip current request due to no cache needs to insert
                if not hit_cache:
                    continue
                # (whole or part tokens) cache hit
                else:
                    if self.load_chunk_cache:
                        chunk_pos_ids = ret[-1]
                        # re-rope
                        # shift delta position for each chunk k cache
                        logger.debug("apply re-rope for chunk kv cache loading...")
                        keys = _apply_k_cache_rerope(keys,
                                                     chunk_pos_ids,
                                                     model_executable,
                                                     start_layer,
                                                     end_layer)

                    # update the end position based on how many tokens are cached.
                    end_pos = start_pos + num_computed_tokens

                    # calculate cached block indices and offsets
                    slot_indices = torch.arange(start_pos, end_pos,
                                                device=input_tokens_tensor.device)
                    slot_mapping_cached = slot_mapping[idx].index_select(0, slot_indices)
                    block_indices_cached = torch.div(slot_mapping_cached,
                                                     self.block_size, rounding_mode="floor")
                    block_offsets_cached = torch.fmod(slot_mapping_cached, self.block_size)

                    # put received KV caches into paged memory
                    for i in range(start_layer, end_layer):
                        kv_cache = kv_caches[i - start_layer]
                        key_cache, value_cache = kv_cache[0], kv_cache[1]
                        key = keys[i - start_layer].to(key_cache.device)
                        value = values[i - start_layer].to(value_cache.device)
                        key_cache = ops.insert_or_update_cache(key,
                                                               key_cache,
                                                               block_indices_cached,
                                                               block_offsets_cached)
                        value_cache = ops.insert_or_update_cache(value,
                                                                 value_cache,
                                                                 block_indices_cached,
                                                                 block_offsets_cached)

                    hidden_or_intermediate_states_for_one_req.append(hidden)

        if not bypass_model_exec:
            # Some of the KV cache is not retrieved
            # so we need to adjust model_input and redo the forwarding.
            logger.debug(
                "[rank%d]: Failed to receive all KVs and hidden "
                "states, redo model forwarding.", torch.distributed.get_rank())
            # logger.debug(f"model_input before: {model_input}")
            self.original_input_tokens_t = model_input.input_tokens
            self.original_slot_mapping = model_input.attn_metadata.slot_mapping
            rebuilt_model_input = build_partial_prefill_input(
                model_input,
                input_tokens_list,
                num_computed_tokens_list,
                start_pos_list,
                slot_mapping,
                device=input_tokens_tensor.device,
                block_size=self.block_size,
            )
            model_input = rebuilt_model_input
            hidden_or_intermediate_states = None
            # logger.debug(f"model_input after: {model_input}")

        else:
            logger.debug(
                "[rank%d]: Successfully received all KVs and hidden "
                "states, skip model forwarding.", torch.distributed.get_rank())
            hidden_or_intermediate_states = torch.cat(
                hidden_or_intermediate_states_for_one_req, dim=0)

        return hidden_or_intermediate_states, bypass_model_exec, model_input

def _apply_k_cache_rerope(k_cache, chunk_position_ids, model_executable, start_layer, end_layer):
    from habana_frameworks.torch.hpex.kernels import (
            RotaryPosEmbeddingMode, apply_rotary_pos_emb)

    rope = _get_model_executable_rope(model_executable)
    seq_len = k_cache.size(1)
    position_ids = torch.arange(0, seq_len, device=chunk_position_ids.device).view(1, -1)
    position_ids = position_ids - chunk_position_ids
    logger.debug(f"position_ids {position_ids}")
    rope.prepare_cos_sin(position_ids)
    is_neox_style = rope.is_neox_style
    sin = rope.sin
    cos = rope.cos
    num_tokens = seq_len
    head_size = rope.head_size
    rotary_dim = rope.rotary_dim

    # HPU RoPE kernel requires hidden dimension for cos and sin to be equal
    # to query hidden dimension, so the original tensors need to be
    # expanded
    # GPT-NeoX kernel requires position_ids = None, offset, mode = BLOCKWISE
    # and expansion of cos/sin tensors via concatenation
    # GPT-J kernel requires position_ids = None, offset = 0, mode = PAIRWISE
    # and expansion of cos/sin tensors via repeat_interleave
    rope_mode: RotaryPosEmbeddingMode
    if is_neox_style:
        rope_mode = RotaryPosEmbeddingMode.BLOCKWISE
    else:
        rope_mode = RotaryPosEmbeddingMode.PAIRWISE
    for i in range(start_layer, end_layer):
        key = k_cache[i - start_layer].to("hpu")
        key_shape = key.shape
        key = key.view(num_tokens, -1, head_size)
        key_rot = key[..., :rotary_dim]
        key_pass = key[..., rotary_dim:]
        key_rot = apply_rotary_pos_emb(key_rot, cos, sin, None, 0, rope_mode)
        key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
        k_cache[i - start_layer] = key

    return k_cache

def _get_model_executable_rope(model_executable):
    model_name = model_executable.layer_names['model_name']
    layers_name = model_executable.layer_names['layers_name']
    attn_name = model_executable.layer_names['attn_name']
    rope_name = model_executable.layer_names['rope_name']

    base_model = getattr(model_executable.model, model_name)
    first_model_layer = getattr(base_model, layers_name)[0]
    attention_layer = getattr(first_model_layer, attn_name)
    rope = getattr(attention_layer, rope_name)
    return rope

def _get_indices_and_offsets(slot_mapping, block_size):
    sm_flatten = slot_mapping.flatten()
    indices = torch.div(sm_flatten, block_size, rounding_mode="floor")
    # only for prompt now
    indices = indices.unflatten(0, (-1, block_size))[:, 0]
    offsets = None

    return indices, offsets

def build_partial_prefill_input(
    model_input: "ModelInputForHPUWithSamplingMetadata",
    input_tokens_list: List[torch.Tensor],
    num_computed_tokens_list: List[int],
    start_pos_list: List[int],
    slot_mapping: torch.Tensor,
    device: torch.device,
    block_size: int = 128
) -> "ModelInputForHPUWithSamplingMetadata":
    """
    Helper function to rebuild the model input for the current request.
    Goal: avoid running redundant prefill on those tokens that already has KV
    caches received.

    use prompt_attention_with_context to implement forward (like prefix-caching)
    1. organize computed_block_nums based on fetched kv cache (cached_num_tokens // block_size).
       use complete blocks and ignore partial blocks (padding effect in hpu_graphs and position_ids)
    2. redo simplified _prepare_prompt process and organize some important model_input member:
       input_tokens (with padding), prefix_block_tables, context_lens, slot_mapping,
       input_positions, etc. attn_bias (mask) will be updated in HpuModelAdapter before forward.
    3. no batch padding there since `build_partial_prefill_input` helps get all hidden_states
       for first_token/next_token generation. If a batch receives all kv caches, we will make
       its seq_len to 1 rather than performing batch reduction.
    """

    # no request receive any kv caches
    logger.debug(f"num_computed_tokens_list {num_computed_tokens_list}")
    if all(nt ==0 for nt in num_computed_tokens_list[:model_input.real_batch_size]):
        return model_input

    from vllm.worker.hpu_model_runner import (_PAD_SLOT_ID,
                                              _PAD_BLOCK_ID)
    from vllm_hpu_extension.bucketing import (HPUBucketingGlobalState,
                                              find_bucket)

    original_bs, padded_seq_len = model_input.input_tokens.size()
    # same
    rebuilt_seq_lens = model_input.seq_lens
    assert (len(rebuilt_seq_lens) == original_bs)

    # context_lens
    new_num_computed_tokens_list = []
    for idx in range(len(input_tokens_list)):
        token_tensor = input_tokens_list[idx]
        num_computed_token = num_computed_tokens_list[idx]
        # currently attention kernel cannot handle the case where there is 0
        # re-calculate 1 tokens for getting batched hidden_states
        if num_computed_token == token_tensor.size(-1):
            new_num_computed_tokens_list.append(num_computed_token -1)
        else:
            new_num_computed_tokens_list.append(num_computed_token)
    # number of computed tokens which are in complete blocks
    # for example, 257 --> 256 (2*128) if block_size=128
    complete_num_blocks_list = [nt // block_size for nt in new_num_computed_tokens_list]
    rebuilt_context_lens = [nb * block_size for nb in complete_num_blocks_list]
    has_context = any(rebuilt_context_lens)

    # remained seq_lens (query_lens)
    assert(len(rebuilt_seq_lens) == len(rebuilt_context_lens))

    rebuilt_query_lens = [rebuilt_seq_lens[i] - rebuilt_context_lens[i]
                           for i in range(original_bs)]
    rebuilt_max_query_len = max(rebuilt_query_lens)
    rebuilt_sum_query_len = sum(rebuilt_query_lens)
    rebuilt_max_prompt_len = max(find_bucket(rebuilt_max_query_len,
                        HPUBucketingGlobalState().prompt_seq_bucket_cfg),
                        block_size)
    logger.debug(f"HPUBucketingGlobalState {HPUBucketingGlobalState().prompt_seq_bucket_cfg}")
    logger.debug(f"rebuilt_max_prompt_len {rebuilt_max_prompt_len}")

    rebuilt_input_tokens_tensor = torch.zeros((original_bs, rebuilt_max_prompt_len),
                                        dtype=torch.long,
                                        device=device)

    rebuilt_input_positions_tensor = torch.zeros((original_bs, rebuilt_max_prompt_len),
                                        dtype=torch.long,
                                        device=device)

    rebuilt_context_lens_tensor = torch.tensor(rebuilt_context_lens,
                                        dtype=torch.long,
                                        device=device)

    rebuilt_seq_lens_tensor = torch.tensor(rebuilt_seq_lens,
                                           dtype=torch.long,
                                           device=device)

    rebuilt_slot_mapping_tensor = torch.full((original_bs, rebuilt_max_prompt_len),
                                             _PAD_SLOT_ID,
                                             dtype=torch.long,
                                             device=device)

    if has_context:
        prefix_block_list_tensor = torch.full((original_bs, max(complete_num_blocks_list)),
                                               _PAD_BLOCK_ID,
                                               dtype=torch.long,
                                               device=device)
    else:
        prefix_block_list_tensor = None

    block_indices, _ = _get_indices_and_offsets(model_input.attn_metadata.slot_mapping, block_size)
    block_indices = block_indices.reshape(original_bs, -1)
    # NOTE: block_offsets is None in prompt attention
    # check it after chunk-prefill enabled
    assert  model_input.attn_metadata.block_offsets is None

    for idx in range(len(input_tokens_list)):
        token_tensor = input_tokens_list[idx]
        num_computed_token = rebuilt_context_lens[idx]
        # always 0 in hpu
        start_pos = start_pos_list[idx]
        num_token = rebuilt_seq_lens[idx]
        assert num_token == token_tensor.size(-1)

        # update input_tokens, input_positions, slot_mapping
        fetch_indices = torch.arange(start_pos + num_computed_token, start_pos + num_token,
                                     device=device)
        update_indices = torch.arange(0, rebuilt_query_lens[idx], device=device)

        rebuilt_input_tokens_tensor[idx].index_copy_(-1, update_indices,
                                         token_tensor.index_select(-1, fetch_indices))

        # NOTE(woosuk): Here we assume that the first token in the prompt
        # is always the first token in the sequence.
        # input_positions.append(list(range(context_len, seq_len)))
        rebuilt_input_positions_tensor[idx].index_copy_(-1, update_indices,
            model_input.input_positions[idx].index_select(-1, fetch_indices))

        # Attn metadata-related
        rebuilt_slot_mapping_tensor[idx].index_copy_(-1, update_indices,
            model_input.attn_metadata.slot_mapping[idx].index_select(-1, fetch_indices))

        if has_context:
            prefix_update_indices = torch.arange(0, complete_num_blocks_list[idx], device=device)
            prefix_block_list_tensor[idx].index_copy_(-1, prefix_update_indices,
                block_indices[idx].index_select(-1, prefix_update_indices))

    # will be set in HPUModelAdapter
    rebuilt_block_indices, rebuilt_block_offsets = None, None

    # rebuilt attn_metadata
    rebuilt_attn_metadata = deepcopy(model_input.attn_metadata)
    rebuilt_attn_metadata.block_list = prefix_block_list_tensor.reshape(-1)
    rebuilt_attn_metadata.block_indices = rebuilt_block_indices
    rebuilt_attn_metadata.block_offsets = rebuilt_block_offsets
    rebuilt_attn_metadata.seq_lens_tensor = rebuilt_seq_lens_tensor
    rebuilt_attn_metadata.context_lens_tensor = rebuilt_context_lens_tensor
    rebuilt_attn_metadata.num_prefill_tokens = rebuilt_sum_query_len
    rebuilt_attn_metadata.slot_mapping = rebuilt_slot_mapping_tensor

    # rebuilt sampling_metadata
    rebuilt_sampling_metadata = deepcopy(model_input.sampling_metadata)
    for idx, q_len in enumerate(rebuilt_query_lens):
        if rebuilt_sampling_metadata.seq_groups is not None:
            rebuilt_sampling_metadata.seq_groups[idx].query_len = q_len

    paddings = [rebuilt_max_query_len - q for q in rebuilt_query_lens]
    paddings = [0] + paddings[:-1]
    paddings = list(itertools.accumulate(paddings))
    rebuilt_selected_token_indices = [(ql + p - 1) for ql, p in zip(rebuilt_query_lens, paddings)]
    rebuilt_sampling_metadata.selected_token_indices = torch.tensor(
        rebuilt_selected_token_indices,
        dtype=model_input.sampling_metadata.selected_token_indices.dtype,
        device=device,
    )

    # import here to avoid circular import.
    from vllm.worker.hpu_model_runner import ModelInputForHPUWithSamplingMetadata
    rebuilt_model_input = ModelInputForHPUWithSamplingMetadata(
        input_tokens=rebuilt_input_tokens_tensor,
        input_positions=rebuilt_input_positions_tensor,
        seq_lens=rebuilt_seq_lens,
        query_lens=rebuilt_query_lens,
        lora_mapping=model_input.lora_mapping,
        lora_requests=model_input.lora_requests,
        attn_metadata=rebuilt_attn_metadata,
        multi_modal_kwargs=model_input.multi_modal_kwargs,
        real_batch_size=model_input.real_batch_size,
        batch_size_padded=model_input.batch_size_padded,
        virtual_engine=model_input.virtual_engine,
        lora_ids=model_input.lora_ids,
        async_callback=model_input.async_callback,
        is_first_multi_step=model_input.is_first_multi_step,
        is_last_step=model_input.is_last_step,
        sampling_metadata=rebuilt_sampling_metadata,
        is_prompt=model_input.is_prompt,
    )

    return rebuilt_model_input
