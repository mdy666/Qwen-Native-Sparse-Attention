import sys
sys.path.append('/sharedata/mdy/code/repo/Megatron-LM')
import torch
from megatron.core.datasets import gpt_dataset
from megatron.core.datasets import indexed_dataset
from transformers import AutoTokenizer
from megatron.training.tokenizer.tokenizer import _HuggingFaceTokenizer
from transformers import Qwen2ForCausalLM
from megatron.core.datasets.utils import Split
import numpy as np
import random

class DS(gpt_dataset.GPTDataset):
    def __getitem__(self, idx):
        """Abstract method implementation

        Args:
            idx (Optioal[int]): The index into the dataset

        Returns:
            Dict[str, torch.Tensor]: The sample information wrapped in a dictionary
        """
        if idx is None:
            # Batch padding sequence so the index does not matter
            text, _ = self._query_document_sample_shuffle_indices(0)
        else:
            text, _ = self._query_document_sample_shuffle_indices(idx)

        text = torch.from_numpy(text).long()
        if self.config.add_extra_token_to_sequence:
            tokens = text[:-1].contiguous()
            labels = text[1:].contiguous()
        else:
            tokens = text
            labels = torch.roll(text, shifts=-1, dims=0)
            labels[-1] = self._pad_token_id

        position_ids = torch.arange(0, len(tokens))
        return {"input_ids": tokens,
                "labels": labels,
                "position_ids": position_ids}


    def concat_bacth(self, bacth):
        return {"input_ids": torch.stack([b['input_ids'] for b in bacth]),
                "labels": torch.stack([b['labels'] for b in bacth]),
                "position_ids": torch.stack([b['position_ids'] for b in bacth])
                }

def build_dataset(data_prefix, tokenizer_path, seq_len=4096, num_samples=None) -> DS:
    index_ds = indexed_dataset.IndexedDataset(data_prefix)
    tokenizer = _HuggingFaceTokenizer(tokenizer_path)
    config = gpt_dataset.GPTDatasetConfig(tokenizer=tokenizer, random_seed=42, sequence_length=seq_len,
                                        reset_attention_mask=False, create_attention_mask=False, 
                                        reset_position_ids=False, eod_mask_loss=False)
    indices = np.arange(0, index_ds.index.sequence_count)
    if num_samples is None:
        # 转list，numpy是int32，容易overflow
        num_samples = sum(index_ds.index.sequence_lengths.tolist()) // seq_len
    ds = DS(index_ds, data_prefix, indices, num_samples, Split.train, config)
    return ds