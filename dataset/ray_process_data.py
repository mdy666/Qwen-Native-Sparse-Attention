# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Processing large data for pretraining."""
import argparse
import math
import json
import os
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
import time
import gzip
import glob
import torch
import numpy as np
import multiprocessing
try:
    import nltk
    from nltk.tokenize.punkt import PunktLanguageVars
    nltk_available = True
except ImportError:
    PunktLanguageVars = object  # Fallback to the built-in object class
    nltk_available = False

from megatron.training.tokenizer import build_tokenizer
from megatron.training.arguments import _add_tokenizer_args
from megatron.core.datasets import indexed_dataset

import pyarrow.parquet as pq
import ray


# https://stackoverflow.com/questions/33139531/preserve-empty-lines-with-nltks-punkt-tokenizer
class CustomLanguageVars(PunktLanguageVars):

    _period_context_fmt = r"""
        \S*                          # some word material
        %(SentEndChars)s             # a potential sentence ending
        \s*                       #  <-- THIS is what I changed
        (?=(?P<after_tok>
            %(NonWord)s              # either other punctuation
            |
            (?P<next_tok>\S+)     #  <-- Normally you would have \s+ here
        ))"""

class IdentitySplitter(object):
    def tokenize(self, *text):
        return text


class Encoder(object):
    def __init__(self, args):
        self.args = args

    def initializer(self):
        # Use Encoder class as a container for global data
        Encoder.tokenizer = build_tokenizer(self.args)
        if self.args.split_sentences:
            if not nltk_available:
                print("NLTK is not available to split sentences.")
                exit()
            if os.environ.get("NLTK_DATA"):
                library = os.path.join(os.environ.get("NLTK_DATA"), "tokenizers", "punkt", f"{self.args.lang}.pickle")
                url = f"file:{library}"
            else:
                library = os.path.join("tokenizers", "punkt", f"{self.args.lang}.pickle")
                url = f"nltk:{library}"
            splitter = nltk.load(url)
            if self.args.keep_newlines:
                # this prevents punkt from eating newlines after sentences
                Encoder.splitter = nltk.tokenize.punkt.PunktSentenceTokenizer(
                    train_text = splitter._params,
                    lang_vars = CustomLanguageVars())
            else:
                Encoder.splitter = splitter

        else:
            Encoder.splitter = IdentitySplitter()

    def split(self, json_line):
        data = json.loads(json_line)
        output = {}
        for key in self.args.json_keys:
            text = data[key]
            max_len = 1000000
            tokens_list = [Encoder.splitter.tokenize(text[i:i+max_len]) for i in range(0, len(text), max_len)]
            output[key] = [tokens for partial in tokens_list for tokens in partial]
        return json.dumps(output), len(json_line)

    def encode(self, json_line):
        data = json_line
        ids = {}
        lens = {}
        for key in self.args.json_keys:
            text = data[key]
            if isinstance(text, list):
                sentences = text
            else:
                sentences = [text]
            doc_ids = []
            sentence_lens = []
            for sentence in sentences:
                sentence_ids = Encoder.tokenizer.tokenize(sentence)
                if len(sentence_ids) > 0:
                    doc_ids.extend(sentence_ids)
                    sentence_lens.append(len(sentence_ids))
            if len(doc_ids) > 0 and self.args.append_eod:
                doc_ids.append(Encoder.tokenizer.eod)
                sentence_lens[-1] += 1
            ids[key] = doc_ids
            lens[key] = sentence_lens
        return ids, lens, len(json_line)
    
    def batch_encode(self, batch):
        output = []
        for key in self.args.json_keys:
            for idx, text in enumerate(batch[key]):
                if idx >= len(output):
                    output.append([{}, {}, 0])
                ids, lens, _ = output[idx]
                if isinstance(text, list):
                    sentences = text
                else:
                    sentences = [text]
                output[idx][-1] += sum([len(text) for text in sentences])
                doc_ids = []
                sentence_lens = []
                for sentence in sentences:
                    sentence_ids = Encoder.tokenizer.tokenize(sentence)
                    if len(sentence_ids) > 0:
                        doc_ids.extend(sentence_ids)
                        sentence_lens.append(len(sentence_ids))
                if len(doc_ids) > 0 and self.args.append_eod:
                    doc_ids.append(Encoder.tokenizer.eod)
                    sentence_lens[-1] += 1
                ids[key] = doc_ids
                lens[key] = sentence_lens
        return output
    

class Partition(object):
    def __init__(self, args, workers):
        self.args = args
        self.workers = workers

    def print_processing_stats(self, count, proc_start, total_bytes_processed):
        if count % self.args.log_interval == 0:
            current = time.time()
            elapsed = current - proc_start
            mbs = total_bytes_processed/elapsed/1024/1024
            print(f"Processed {count} documents",
                  f"({count/elapsed} docs/s, {mbs} MB/s).",
                  file=sys.stderr)

    def split_sentences(self, file_name):
        input_file_name, output_file_name = file_name
        print("Opening", input_file_name)
        fin = open(input_file_name, 'r', encoding='utf-8')
        fout = open(output_file_name, 'w')

        encoder = Encoder(self.args)
        pool = multiprocessing.Pool(self.workers, initializer=encoder.initializer)
        split_docs = pool.imap(encoder.split, fin, 32)

        proc_start = time.time()
        total_bytes_processed = 0
        for i, (doc, bytes_processed) in enumerate(split_docs, start=1):
            total_bytes_processed += bytes_processed
            fout.write(doc + "\n")
            self.print_processing_stats(i, proc_start, total_bytes_processed)

        fin.close()
        fout.close()

    def process(self, output_prefix, idx):
        total_files = sorted(glob.glob(self.args.input))[:72]
        part_files = total_files[idx::self.args.partitions]

        print(idx)
        time.sleep(idx) # 一起读取可能有的读取不上，等一下
        if part_files[0].endswith('parquet'):
            ds = ray.data.read_parquet(part_files)
            print(f'这是 {idx}')
        elif part_files[0].endswith('json') or part_files[0].endswith('jsonl'):
            ds = ray.data.read_json(part_files)
        else:
            assert False, 'only support json or parquet files'
        
        # time.sleep(3)
        # time.sleep(1000)
        startup_start = time.time()
        encoder = Encoder(self.args)
        tokenizer = build_tokenizer(self.args)
        pool = multiprocessing.Pool(self.workers, initializer=encoder.initializer)
        encoded_docs = pool.imap(encoder.encode, ds.iter_rows(), 32)

        level = "document"
        if self.args.split_sentences:
            level = "sentence"

        output_bin_files = {}
        output_idx_files = {}
        builders = {}

        for key in self.args.json_keys:
            output_bin_files[key] = "{}_{}_{}.bin".format(output_prefix,
                                                          key, level)
            output_idx_files[key] = "{}_{}_{}.idx".format(output_prefix,
                                                          key, level)
            builders[key] = indexed_dataset.IndexedDatasetBuilder(
                output_bin_files[key],
                dtype=indexed_dataset.DType.optimal_dtype(tokenizer.vocab_size),
            )

        startup_end = time.time()
        proc_start = time.time()
        total_bytes_processed = 0
        print("Time to startup:", startup_end - startup_start)
        for i, (doc, sentence_lens, bytes_processed) in enumerate(encoded_docs, start=1):
            total_bytes_processed += bytes_processed
            for key in doc.keys():
                builders[key].add_document(doc[key], sentence_lens[key])
            self.print_processing_stats(i, proc_start, total_bytes_processed)

        for key in self.args.json_keys:
            builders[key].finalize(output_idx_files[key])

    def batch_process(self, output_prefix, idx):
        total_files = sorted(glob.glob(self.args.input))[:72]
        part_files = total_files[idx::self.args.partitions]

        print(idx)
        time.sleep(idx) # 一起读取可能有的读取不上，等一下
        if part_files[0].endswith('parquet'):
            ds = ray.data.read_parquet(part_files)
            print(f'这是 {idx}')
        elif part_files[0].endswith('json') or part_files[0].endswith('jsonl'):
            ds = ray.data.read_json(part_files)
        else:
            assert False, 'only support json or parquet files'
        
        # time.sleep(3)
        # time.sleep(1000)
        startup_start = time.time()
        encoder = Encoder(self.args)
        tokenizer = build_tokenizer(self.args)
        pool = multiprocessing.Pool(self.workers, initializer=encoder.initializer)
        encoded_docs = pool.imap(encoder.batch_encode, ds.iter_batches(batch_size=64), 32)

        level = "document"
        if self.args.split_sentences:
            level = "sentence"

        output_bin_files = {}
        output_idx_files = {}
        builders = {}

        for key in self.args.json_keys:
            output_bin_files[key] = "{}_{}_{}.bin".format(output_prefix,
                                                          key, level)
            output_idx_files[key] = "{}_{}_{}.idx".format(output_prefix,
                                                          key, level)
            builders[key] = indexed_dataset.IndexedDatasetBuilder(
                output_bin_files[key],
                dtype=indexed_dataset.DType.optimal_dtype(tokenizer.vocab_size),
            )

        startup_end = time.time()
        proc_start = time.time()
        total_bytes_processed = 0
        print("Time to startup:", startup_end - startup_start)
        i = 1
        for batch in encoded_docs:
            for doc, sentence_lens, bytes_processed in batch:
                total_bytes_processed += bytes_processed
                for key in doc.keys():
                    builders[key].add_document(doc[key], sentence_lens[key])
                self.print_processing_stats(i, proc_start, total_bytes_processed)
                i += 1

        for key in self.args.json_keys:
            builders[key].finalize(output_idx_files[key])


def get_args():
    parser = argparse.ArgumentParser()
    parser = _add_tokenizer_args(parser)
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', type=str, required=True,
                       help='Path to input JSON')
    group.add_argument('--json-keys', nargs='+', default=['text'],
                       help='space separate listed of keys to extract from json')
    group.add_argument('--split-sentences', action='store_true',
                       help='Split documents into sentences.')
    group.add_argument('--keep-newlines', action='store_true',
                       help='Keep newlines between sentences when splitting.')
    group = parser.add_argument_group(title='tokenization process')
    group.add_argument('--append-eod', action='store_true',
                       help='Append an <eod> token to the end of a document.')
    group.add_argument('--lang', type=str, default='english',
                       help='Language to use for NLTK-powered sentence splitting.')
    group = parser.add_argument_group(title='output data')
    group.add_argument('--output-prefix', type=str, required=True,
                       help='Path to binary output file without suffix')
    group = parser.add_argument_group(title='runtime')
    group.add_argument('--workers', type=int, required=True,
                       help=('Number of worker processes to launch.'
                             'A good default for fast pre-processing '
                             'is: (workers * partitions) = available CPU cores.'))
    group.add_argument('--partitions', type=int, default=1,
                        help='Number of file partitions')
    group.add_argument('--log-interval', type=int, default=1000,
                       help='Interval between progress updates')
    group.add_argument('--keep-sequential-samples', action='store_true',
                       help='Ensure ordering of samples in .jsonl files is '
                            'preserved when using partitions>1.')
    args = parser.parse_args()
    args.keep_empty = False

    if args.tokenizer_type.lower().startswith('bert') and not args.split_sentences:
        print("Are you sure you don't want to split sentences?")

    # some default/dummy values for the tokenizer
    args.rank = 1
    args.make_vocab_size_divisible_by = 128
    args.tensor_model_parallel_size = 1
    args.vocab_extra_ids = 0

    return args


def get_file_name(args, file_id):
    file_name, extension = os.path.splitext(args.input)
    extension = '.json'
    input_file_name = file_name + "_" + str(file_id) + extension
    sentence_split_file = file_name + "_ss_" + str(file_id) + extension
    output_prefix = args.output_prefix + "_" + str(file_id)
    file_names = {
        'partition': input_file_name,
        'sentence_split': sentence_split_file,
        'output_prefix': output_prefix}
    return file_names


def check_files_exist(in_ss_out_names, key, num_partitions):
    for i in range(num_partitions):
        if not os.path.exists(in_ss_out_names[i][key]):
            return False
    return True

def check_and_create_parent_directory(input_path):
    """
    检查输入路径的上一层目录是否存在，如果不存在则创建该文件夹。

    参数:
        input_path (str): 输入路径，例如 "/sharedata/mdy/data/pretrain/tmp/fineweb_edu_zh"
    """
    # 获取上一层目录
    parent_directory = os.path.dirname(input_path)
    
    # 检查上一层目录是否存在
    if not os.path.exists(parent_directory):
        print(f"上一层目录 '{parent_directory}' 不存在，正在创建...")
        os.makedirs(parent_directory)  # 创建文件夹
        print(f"目录 '{parent_directory}' 创建成功。")
    else:
        print(f"上一层目录 '{parent_directory}' 已存在。")
        

def main():
    start_time = time.time()
    args = get_args()

    assert args.workers % args.partitions == 0
    partition = Partition(args, args.workers//args.partitions)

    check_and_create_parent_directory(args.output_prefix)

    per_part_save_path = []
    if args.partitions == 1:
        per_part_save_path.append(args.output_prefix)
    else:
        for idx in range(args.partitions):
            per_part_save_path.append(args.output_prefix + f'_part_{str(idx).zfill(2)}')

    processes = []
    for idx, name in enumerate(per_part_save_path):
        p = multiprocessing.Process(target=partition.batch_process,
                                    args=(name, idx))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # tasks = [partition.process_json_file.remote(name, idx) for idx, name in enumerate(per_part_save_path)]
    # ray.get(tasks)


    if args.partitions == 1:
        return


    level = "document"

    output_bin_files = {}
    output_idx_files = {}
    builders = {}
    tokenizer = build_tokenizer(args)

    for key in args.json_keys:
        output_bin_files[key] = "{}_{}_{}.bin".format(args.output_prefix,
                                                      key, level)
        output_idx_files[key] = "{}_{}_{}.idx".format(args.output_prefix,
                                                      key, level)
        builders[key] = indexed_dataset.IndexedDatasetBuilder(
            output_bin_files[key],
            dtype=indexed_dataset.DType.optimal_dtype(tokenizer.vocab_size),
        )

        for name in per_part_save_path:
            parition_output_prefix = name
            full_partition_output_prefix = "{}_{}_{}".format(parition_output_prefix,
                                                             key, level)
            builders[key].add_index(full_partition_output_prefix)
            print(full_partition_output_prefix)
            os.remove(f'{full_partition_output_prefix}.bin')
            os.remove(f'{full_partition_output_prefix}.idx')  
        builders[key].finalize(output_idx_files[key])

    print(f'用时: {time.time() - start_time} s')


if __name__ == '__main__':

    main()

