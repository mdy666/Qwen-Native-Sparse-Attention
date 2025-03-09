# 介绍

- 将 **NSA (Native Sparse Attention)** 应用于 **Qwen2.5** 中
- 论文链接：[NSA Paper](https://arxiv.org/pdf/2502.11089)

---

# 启动命令

在 `train.sh` 中配置好其它参数后，使用以下命令进行启动：

```bash
# base-bf16
bash train.sh --deepspeed

# base-fp8
bash train.sh --deepspeed --fp8 --fp8-pattern proj

# nsa-bf16
bash train.sh --deepspeed --nsa

# nsa-fp8
bash train.sh --deepspeed --nsa --fp8 --fp8-pattern proj
```

---

# 测试结果

## 训练损失 (Training Loss)

- **问题**：配置文件写错了（QWQ），本来要训练 Qwen 3B，但模型层数改错了，变成了 1.5B。具体日志使用tensorboard查看log文件夹
  
  ![Training Loss](./log/imgs/train_loss.png)

## NSA 前向传播 (NSA Forward)

- **说明**：NSA 是端到端的时间，输入 `qkv`，输出 `combine-o`，包括 `compress_attn`、`select_attn`、`window_attn` 和 `combine`。
  
  ![NSA Forward](./log/imgs/nsa_fwd.png)
```bash
forward:
         n        NSA  Triton-FA2         FA2         FA3
0   4096.0   1.746017    1.149936    0.879112    0.454097
1   8192.0   3.766880    4.257213    3.261264    1.738003
2  16384.0   8.446412   16.563154   12.620720    6.847600
3  32768.0  20.519409   66.391747   49.222591   26.815157
4  65536.0  56.798977  273.811615  195.105530  107.702240
```

## NSA 反向传播 (NSA Backward)

  ![NSA Backward](./log/imgs/nsa_bwd.png)
```bash
backward:
         n         NSA  Triton-FA2         FA2         FA3
0   4096.0    4.491698    3.943456    2.477263    1.379212
1   8192.0    9.415763   14.986874    8.863830    5.015210
2  16384.0   20.790792   58.587681   33.741039   18.849483
3  32768.0   49.167553  235.818939  136.086914   74.696991
4  65536.0  127.783165  933.758667  533.677673  294.244171

```

---

# 具体文件说明

## `nsa_attention` 文件夹

- 该文件夹中 `compress_attn` 和 `select_attn` 包含多个版本，`v1` 是我最初的版本。
- **为什么会有 `v2` 版本？**  
  我使用了大佬们开发的 NSA 仓库（[Native Sparse Attention](https://github.com/fla-org/native-sparse-attention)），发现同样代码，他们的 `select_attn` 的 `forward` 比我快了一倍。唯一区别是我的 Triton 代码里都是自己去做 `ptrs`，没有使用 `tl.make_block_ptr` 函数。`v2` 版本是所有的 `attention` 相关的 kernel 都使用 `tl.make_block_ptr` 去生成指针。具体对比可以在 `精度和性能测试.ipynb` 文件中查看：
  - `compress_attn` 的 `v1` 和 `v2` 差不多，`v1`略快一点点。
  - `select_attn` 的 `v1` 的 `forward` 比 `v2` 慢了一倍。
  - `select_v3` 是 `fwd` 和 `bwd_dq` 使用 `tl.make_block_ptr` 去制作指针，其他与 `v1` 保持不变。

## `triton_kernel` 文件夹

- 替换 `transformers` 中一些算子，效率更高效。

## `fp8` 文件夹

- 应用 **DeepSeek** 开发的 **DeepGemm** 到训练中。

## `dataset` 文件夹

- 使用 **Megatron GPT Dataset** 读取数据，可替换。  
  **注意**：`label` 已经是 `shift` 之后的了。
