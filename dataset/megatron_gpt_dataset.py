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
import math

class SingleDS(gpt_dataset.GPTDataset):
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
    
class MultiDS(torch.utils.data.Dataset):
    def __init__(self, data_prefix_list, config, num_samples=None):
        super().__init__()
        
        try:
            float(data_prefix_list[0])
            self.weight = [float(w) for w in data_prefix_list[::2]]
            self.data_prefix_list =  data_prefix_list[1::2]
        except:
            self.weight = None
            self.data_prefix_list =  data_prefix_list

        self.config = config
        self.seq_len = config.sequence_length
        self.index_ds_list = [indexed_dataset.IndexedDataset(data_prefix) for data_prefix in self.data_prefix_list]
        self.num_samples_per_epoch_list = np.array([sum(index_ds.index.sequence_lengths.tolist()) // self.seq_len for index_ds in self.index_ds_list], dtype=np.int64)

        if self.weight is None:
            self.weight = self.num_samples_per_epoch_list / self.num_samples_per_epoch_list.sum()

        if num_samples is None:
            num_samples = int(self.num_samples_per_epoch_list.sum())
        
        self.num_samples_per_ds_list = [math.ceil(w * num_samples) for w in self.weight]
        self.num_samples = sum(self.num_samples_per_ds_list)

        self.ds_idx = None
        self.data_idx = None
        self.sample_idx = None
        self.dataset_list = []

        self.init_sample_policy()

        
    def __len__(self):
        return self.num_samples
    
    def init_sample_policy(self):
        sample_ds_idx = []
        sample_data_idx = []
        for i in range(len(self.data_prefix_list)):
            sample_ds_idx.extend([i] * self.num_samples_per_ds_list[i])
            sample_data_idx.extend(list(range(0, self.num_samples_per_ds_list[i])))

            indices = np.arange(0, self.index_ds_list[i].index.sequence_count)
            self.dataset_list.append(SingleDS(self.index_ds_list[i], self.data_prefix_list[i], indices, self.num_samples_per_ds_list[i], Split.train, self.config))

        self.ds_idx = np.array(sample_ds_idx, dtype=np.int32)
        self.data_idx = np.array(sample_data_idx, dtype=np.int32)
        np.random.seed(self.config.random_seed)
        self.sample_idx = np.random.permutation(np.arange(0, self.num_samples, dtype=np.int32))

    def __getitem__(self, index):
        idx = self.sample_idx[index]
        ds_idx = self.ds_idx[idx]
        data_idx = self.data_idx[idx]
        output = {"dataset": self.data_prefix_list[ds_idx]}
        output.update(self.dataset_list[ds_idx][data_idx])
        return output
    
    def concat_bacth(self, bacth):
        return {"input_ids": torch.stack([b['input_ids'] for b in bacth]),
                "labels": torch.stack([b['labels'] for b in bacth]),
                "position_ids": torch.stack([b['position_ids'] for b in bacth])
                }
        

def build_dataset(data_prefix, tokenizer_path, seq_len=4096, seed=42, num_samples=None) -> torch.utils.data.Dataset:
    tokenizer = _HuggingFaceTokenizer(tokenizer_path)
    config = gpt_dataset.GPTDatasetConfig(tokenizer=tokenizer, random_seed=seed, sequence_length=seq_len,
                                        reset_attention_mask=False, create_attention_mask=False, 
                                        reset_position_ids=False, eod_mask_loss=False)
    if len(data_prefix) == 1:
        data_prefix = data_prefix[0]
        index_ds = indexed_dataset.IndexedDataset(data_prefix)
        indices = np.arange(0, index_ds.index.sequence_count)
        if num_samples is None:
            # 转list，numpy是int32，容易overflow
            num_samples = sum(index_ds.index.sequence_lengths.tolist()) // seq_len
        ds = SingleDS(index_ds, data_prefix, indices, num_samples, Split.train, config)
    else:
        ds = MultiDS(data_prefix, config, num_samples)
    return ds