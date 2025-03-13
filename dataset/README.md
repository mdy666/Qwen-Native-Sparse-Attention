# Megatron Dataset

用于大规模预训练，直接从内存中读取对应数据，不用一次加载进来

# 使用方法

## step 1

```bash
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
python setup.py build_ext --inplace
```
## step 2

将**ray_process_data.py**放到**Megatron-LM/tools**下，将**ray_process_data.sh**放到**Megatron-LM**文件夹下

## step 3

从huggingface或者model-scope上下载大量json或者parquet数据

## step 4

```bash
# 配置好参数，输入文本路径，选择好要处理的字段，一般是“text”
bash ray_process_data.sh
```

## step 5

```python
from dataset.megatron_gpt_dataset import build_dataset
tokenizer = '/sharedata/mdy/models/SauerkrautLM-Mixtral-8x7B-Instruct'


# 单个数据集
data_prefix = ["/sharedata/mdy/data/pretrain/mixtral/fineweb_edu_en_text_document"]

# 多个数据集，每个数据集有权重
data_prefix = """0.5 /sharedata/mdy/data/pretrain/mixtral/fineweb_edu_en_text_document \
0.5 /sharedata/mdy/data/pretrain/mixtral/fineweb_edu_zh_text_document""".split(' ')

# 多个数据集，每个数据集没有权重
data_prefix = """/sharedata/mdy/data/pretrain/mixtral/fineweb_edu_en_text_document \
/sharedata/mdy/data/pretrain/mixtral/fineweb_edu_zh_text_document""".split(' ')

ds = build_dataset(data_prefix, tokenizer_path=tokenizer, seed=888, num_samples=1000)
```

# BUG

```python
# Megatron-LM/megatron/core/datasets/indexed_dataset.py中的804行，改如下
# 不然np.int32会发生数值溢出
def add_index(self, path_prefix: str) -> None:
    """Add an entire IndexedDataset to the dataset

    Args:
        path_prefix (str): The index (.idx) and data (.bin) prefix
    """
    # Concatenate index
    index = _IndexReader(get_idx_path(path_prefix), multimodal=self.multimodal)
    assert index.dtype == self.dtype

    offset = len(self.sequence_lengths)
    # self.sequence_lengths.extend(index.sequence_lengths)
    # self.document_indices.extend((offset + index.document_indices)[1:])

    self.sequence_lengths.extend(index.sequence_lengths.tolist())
    self.document_indices.extend((offset + index.document_indices)[1:].tolist())

    if self.multimodal:
        self.sequence_modes.extend(index.sequence_modes)

    # Concatenate data
    with open(get_bin_path(path_prefix), "rb") as f:
        shutil.copyfileobj(f, self.data_file)
```
