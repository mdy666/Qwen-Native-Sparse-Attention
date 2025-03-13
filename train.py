import time
import argparse
import sys
import os
import json
import random
# 替换常规加速算子
from qwen2.patch_others import trigger

import torch
import torch.distributed as dist
from dataset.megatron_gpt_dataset import build_dataset
from transformers import TrainingArguments, Trainer, AutoModelForCausalLM, AutoConfig, Qwen2ForCausalLM

def print_rank0(*args):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(*args)
    else:
        print(*args)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nsa', action='store_true')
    parser.add_argument('--fp8', action='store_true')
    parser.add_argument('--fp8-pattern', nargs='+')
    parser.add_argument('--deepspeed', action='store_true')

    parser.add_argument('--model-config',type=str, default='./qwen2/config1.5B.json')
    parser.add_argument('--tokenizer',type=str, default='')
    parser.add_argument('--data-path', nargs='+')

    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--log_dir', type=str)
    parser.add_argument('--micro_batch_size', type=int, default=4)
    parser.add_argument('--global_batch_size', type=int, default=256)
    parser.add_argument('--max_seq_len', type=int, default=1024)
    parser.add_argument('--max_steps', type=int, default=10000)
    return parser.parse_args()



if __name__ == '__main__':
    dist.init_process_group(backend='nccl')
    seed = 888
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)



    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device('cuda', rank)
    torch.cuda.set_device(device)
    print_rank0(rank, world_size)
    time.sleep(3)

    args = get_args()
    print_rank0(args)
    assert args.global_batch_size % (args.micro_batch_size * world_size) == 0
    acc_grad_steps = args.global_batch_size // (args.micro_batch_size * world_size)

    # 加入NSA配置
    if args.nsa:
        from qwen2.patch_nsa import trigger

    print_rank0('====================loading datasets=======================') 
    # 用的Magatron中的GPTDataset，可以换成别的。注意：label已经是shift之后的了
    ds = build_dataset(args.data_path, args.tokenizer, args.max_seq_len, seed=42, num_samples=args.max_steps * args.global_batch_size)
    
    print_rank0('====================loading model=======================')
    config = AutoConfig.from_pretrained(args.model_config)
    model = Qwen2ForCausalLM(config).to(device).to(torch.bfloat16)
    # 初始化很慢，可以初始化一次，保存下来，之后用from_pretrained加载
    # model.save_pretrained('/sharedata/mdy/models/nsa-3B')
    # model = AutoModelForCausalLM.from_pretrained('/sharedata/mdy/models/nsa-7B', torch_dtype=torch.bfloat16, device_map='cuda')
    print_rank0(model)
    time.sleep(5)

    # 应用fp8。pattern是个正则列表，module的名字匹配上其中一个则将Linear使用FP8进行运算
    # 例如["proj"] 或者 ["q_proj", "k_proj", "v_proj", "up_proj"]
    if args.fp8:
        from fp8.fp8_gemm import apply_fp8
        print_rank0(args.fp8_pattern)
        apply_fp8(model, pattern=args.fp8_pattern)

    print_rank0('====================start trainging=======================')
    train_args = TrainingArguments(
                            output_dir=args.output_dir,
                            logging_dir=args.log_dir,
                            logging_strategy='steps',
                            logging_steps=1,
                            report_to='tensorboard',

                            per_device_train_batch_size=args.micro_batch_size,
                            # per_device_eval_batch_size=args.batch_size_per_device * 2,
                            gradient_accumulation_steps=acc_grad_steps,
                            
                            # evaluation_strategy='steps',
                            # eval_steps=100,

                            save_strategy='steps',
                            save_steps=1000,
                            save_total_limit=10,
                            save_only_model=False,
                            save_safetensors=True,
                            # num_train_epochs=2,

                            dataloader_num_workers=16,
                            dataloader_prefetch_factor=2,

                            max_steps=args.max_steps,
                            warmup_steps=1000,
                            learning_rate=5e-4,             
                            lr_scheduler_type='cosine',
                            weight_decay=0.01,
                            
                            bf16=True,
                            deepspeed='./qwen2/zero2.json' if args.deepspeed else None,  

                            disable_tqdm=False,  
                            seed=seed
                            )
    
    trainer = Trainer(model=model,
                      args=train_args,
                      data_collator=ds.concat_bacth,
                      train_dataset=ds
                      )
    trainer.train()





