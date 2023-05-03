import logging
import pandas as pd
import os
import pickle

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class Example(object):
    """A single training/test example."""

    def __init__(self,
                 idx,
                 source,
                 target
                 ):
        self.idx = idx
        self.source = source
        self.target = target

def read_examples(filename):
    examples = []
    data = pd.read_csv(filename)
    srcs = data['src'].tolist()
    tgts = data['tgt'].tolist()
    for idx in range(len(srcs)):
        examples.append(
            Example(
                idx=idx,
                source=srcs[idx],
                target=tgts[idx],
            )
        )
    return examples


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 source_mask,
                 target_ids,
                 target_mask
                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.source_mask = source_mask
        self.target_ids = target_ids
        self.target_mask = target_mask

def convert_examples_to_features(examples, tokenizer, max_source_length, max_target_length, stage=None):
    features = []
    for example_index, example in enumerate(examples):
        # source
        source_tokens = tokenizer.tokenize(example.source)[:max_source_length - 2]
        source_tokens = [tokenizer.cls_token] + source_tokens + [tokenizer.sep_token]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        source_mask = [1] * (len(source_tokens))
        padding_length = max_source_length - len(source_ids)
        source_ids += [tokenizer.pad_token_id] * padding_length
        source_mask += [0] * padding_length
        # target
        if stage == "test":
            target_tokens = tokenizer.tokenize("None")
        else:
            target_tokens = tokenizer.tokenize(example.target)[:max_target_length - 2]
        target_tokens = [tokenizer.cls_token] + target_tokens + [tokenizer.sep_token]
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        target_mask = [1] * len(target_ids)
        padding_length = max_target_length - len(target_ids)
        target_ids += [tokenizer.pad_token_id] * padding_length
        target_mask += [0] * padding_length

        features.append(
            InputFeatures(
                example_index,
                source_ids,
                source_mask,
                target_ids,
                target_mask,
            )
        )
    return features


class GPTDataset(Dataset):
    def __init__(self, datafile, tokenizer, block_size=512, mode='train'):

        self.block_size = block_size
        self.mode = mode

        self.inputs = []
        self.token_labels = []

        datas = pd.read_csv(datafile)

        length = len(datas)

        for idx in tqdm(range(length)):
            src = tokenizer.encode(datas["src"][idx])
            tgt = tokenizer.encode(datas["tgt"][idx])

            input_ids, input_labels = self.pad_and_get_mask(src, tgt, tokenizer)
            self.inputs.append(input_ids)
            self.token_labels.append(input_labels)

    def pad_and_get_mask(self, src, tgt, tokenizer):
        if self.mode == 'test':
            tgt = []
            
        while (len(src) + len(tgt) + 2 > self.block_size):
            if (len(tgt) > len(src)):
                tgt = tgt[:-1]
            else:
                src = src[:-1]
        if self.mode == 'train':
            inputs = src + [tokenizer.bos_token_id] + tgt + [tokenizer.eos_token_id]
            labels = [1] * len(src) + [2] * (len(tgt) + 1) + [0]
        else:
            inputs = src + [tokenizer.bos_token_id]
            labels = [1] * len(src) + [2]
            return inputs, labels
        assert len(inputs) <= self.block_size
        pad_len = self.block_size - len(inputs)
        inputs += [tokenizer.pad_token_id] * pad_len
        labels += [0] * pad_len
        assert len(inputs) == len(labels)
        return inputs, labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        return torch.tensor(self.inputs[item]), torch.tensor(self.token_labels[item])