import math
import os
import re

import numpy
import pandas as pd
import torch

# from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, PeftConfig, PeftModel
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import RandomSampler, DataLoader, TensorDataset, SequentialSampler
from tqdm import tqdm
from transformers import (RobertaTokenizer, PLBartConfig, PLBartTokenizer, PLBartForConditionalGeneration, GPTNeoForCausalLM, GPTNeoConfig,
                          RobertaConfig, RobertaModel, GPT2Config, GPT2LMHeadModel, GPT2Tokenizer, CodeGenConfig, CodeGenTokenizer, CodeGenForCausalLM,
                          T5Config, T5ForConditionalGeneration, AdamW, get_linear_schedule_with_warmup)
import logging

from NPGD import NPGD
from VAT import virtual_adversarial_training
from UniModel import Uni_Seq2Seq
from bitfit import modify_with_bitfit
from datasets import read_examples, convert_examples_to_features, GPTDataset
from utils import get_bleu_socre
from EncModel import BERT_Seq2Seq, Beam
# import numpy as np

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'codet5': (T5Config, T5ForConditionalGeneration, RobertaTokenizer),
    'natgen': (T5Config, T5ForConditionalGeneration, RobertaTokenizer),
    'plbart': (PLBartConfig, PLBartForConditionalGeneration, PLBartTokenizer),
    'codebert': (RobertaConfig, RobertaModel, RobertaTokenizer),
    'unixcoder': (RobertaConfig, RobertaModel, RobertaTokenizer),
    'graphcodebert': (RobertaConfig, RobertaModel, RobertaTokenizer),
    'contrabert': (RobertaConfig, RobertaModel, RobertaTokenizer),
    'codegpt-java': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'codegpt-py': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'codegpt-adapter-java': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'codegpt-adapter-py': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'codegen': (CodeGenConfig, CodeGenForCausalLM, CodeGenTokenizer),
    'gpt-neo': (GPTNeoConfig, GPTNeoForCausalLM, GPT2Tokenizer)
}

def get_model_size(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    model_size = sum([numpy.prod(p.size()) for p in model_parameters])
    return "{}M".format(round(model_size / 1e+6))

def build_or_load_gen_model(model_type, model_name_or_path, load_model_path, method_type='enc-dec', max_target_length=None, beam_size=None):
    if(method_type=='enc-dec'):
        config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
        config = config_class.from_pretrained(model_name_or_path)
        tokenizer = tokenizer_class.from_pretrained(model_name_or_path)
        model = model_class.from_pretrained(model_name_or_path)
        logger.info("Finish loading model [%s] from %s", get_model_size(model), model_name_or_path)

        if load_model_path is not None:
            logger.info("Reload model from {}".format(load_model_path))
            model.load_state_dict(torch.load(load_model_path))
        return config, model, tokenizer

    if (method_type == 'enc'):
        config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
        config = config_class.from_pretrained(model_name_or_path)
        tokenizer = tokenizer_class.from_pretrained(model_name_or_path)
        encoder = model_class.from_pretrained(model_name_or_path)
        decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
        decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        model = BERT_Seq2Seq(encoder=encoder, decoder=decoder, config=config,
                        beam_size=beam_size, max_length=max_target_length,
                        sos_id=tokenizer.cls_token_id, eos_id=tokenizer.sep_token_id)
        logger.info("Finish loading model [%s] from %s", get_model_size(model), model_name_or_path)
        if load_model_path is not None:
            logger.info("Reload model from {}".format(load_model_path))
            model.load_state_dict(torch.load(load_model_path))
        return config, model, tokenizer
    if (method_type == 'dec'):
        config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
        config = config_class.from_pretrained(model_name_or_path)
        tokenizer = tokenizer_class.from_pretrained(model_name_or_path, do_lower_case=False, bos_token='<s>',
                                                    eos_token='</s>', pad_token='<pad>', unk_token='<|UNKNOWN|>',
                                                    sep_token='<sep>')
        model = model_class.from_pretrained(model_name_or_path)
        model.resize_token_embeddings(len(tokenizer))
        model.config.bos_token_id = tokenizer.bos_token_id
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        logger.info("Finish loading model [%s] from %s", get_model_size(model), model_name_or_path)
        if load_model_path is not None:
            logger.info("Reload model from {}".format(load_model_path))
            model.load_state_dict(torch.load(load_model_path))
        return config, model, tokenizer

    if (method_type == 'unixcoder'):
        config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
        config = config_class.from_pretrained(model_name_or_path)
        config.is_decoder = True
        config.output_attentions = True
        tokenizer = tokenizer_class.from_pretrained(model_name_or_path)
        encoder = model_class.from_pretrained(model_name_or_path, config = config)
        model = Uni_Seq2Seq(encoder=encoder, decoder=encoder, config=config,
                        beam_size=beam_size, max_length=max_target_length,
                        sos_id=tokenizer.convert_tokens_to_ids(["<mask0>"])[0],
                        eos_id=tokenizer.sep_token_id, pad_id=tokenizer.pad_token_id)
        logger.info("Finish loading model [%s] from %s", get_model_size(model), model_name_or_path)
        if load_model_path is not None:
            logger.info("Reload model from {}".format(load_model_path))
            model.load_state_dict(torch.load(load_model_path))
        return config, model, tokenizer

class Encoder_Decoder():
    def __init__(self, model_type, model_name_or_path, load_model_path, beam_size, max_source_length, max_target_length, method_type='enc-dec', bitfit=False):
        self.model_type = model_type
        self.config, self.model, self.tokenizer = build_or_load_gen_model(model_type, model_name_or_path,
                                                                          load_model_path, method_type, bitfit)
        self.method_type = method_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.beam_size, self.max_source_length, self.max_target_length = beam_size, max_source_length, max_target_length


    def train(self, train_filename, train_batch_size, learning_rate, num_train_epochs, early_stop, task,
              do_eval, eval_filename, eval_batch_size, output_dir, do_eval_bleu, AT = False):

        train_examples = read_examples(train_filename)
        train_features = convert_examples_to_features(train_examples, self.tokenizer, self.max_source_length,
                                                      self.max_target_length, stage='train')
        
        all_source_ids = torch.tensor([f.source_ids for f in train_features], dtype=torch.long)
        all_source_mask = torch.tensor([f.source_mask for f in train_features], dtype=torch.long)
        all_target_ids = torch.tensor([f.target_ids for f in train_features], dtype=torch.long)
        all_target_mask = torch.tensor([f.target_mask for f in train_features], dtype=torch.long)

        train_data = TensorDataset(all_source_ids, all_source_mask, all_target_ids, all_target_mask)

        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                      batch_size=train_batch_size)

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        t_total = len(train_dataloader) // num_train_epochs
        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=1e-8)
        num_train_optimization_steps = num_train_epochs * len(train_dataloader)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=int(t_total * 0.1),
                                                    num_training_steps=num_train_optimization_steps)

        # Start training
        train_example_num = len(train_data)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", train_example_num)
        logger.info("  Batch size = %d", train_batch_size)
        logger.info("  Batch num = %d", math.ceil(train_example_num/train_batch_size))
        logger.info("  Num epoch = %d", num_train_epochs)
        dev_dataset = {}
        global_step, best_bleu, best_loss = 0, -1, 1e6
        count = 0
        npgd = NPGD(self.model)
        K = 3
        for cur_epoch in range(int(num_train_epochs)):
            bar = tqdm(train_dataloader, total=len(train_dataloader), desc="Training")
            nb_tr_examples, nb_tr_steps, tr_loss = 0, 0, 0
            self.model.train()
            for step, batch in enumerate(bar):
                batch = tuple(t.to(self.device) for t in batch)
                source_ids, source_mask, target_ids, target_mask = batch
                outputs = self.model(input_ids=source_ids, attention_mask=source_mask,
                                     labels=target_ids, decoder_attention_mask=target_mask)
                total_loss = outputs.loss
                total_loss.backward()

                if AT == True:
                    npgd.backup_grad()
                    for t in range(K):
                        npgd.attack(emb_name='shared', is_first_attack=(t == 0))
                        if t != K-1:
                            self.model.zero_grad()
                        else:
                            npgd.restore_grad()
                        loss_adv = self.model(input_ids=source_ids, attention_mask=source_mask,
                                     labels=target_ids, decoder_attention_mask=target_mask).loss
                        loss_adv.backward()
                    npgd.restore(emb_name='shared')

                tr_loss += total_loss.item()
                nb_tr_examples += source_ids.size(0)
                nb_tr_steps += 1
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                train_loss = round(tr_loss * 1 / (nb_tr_steps + 1), 4)
                bar.set_description("[{}] Train loss {}".format(cur_epoch, round(train_loss, 3)))
        
            if do_eval==True:
                # Eval model with dev dataset
                eval_examples = read_examples(eval_filename)
                eval_features = convert_examples_to_features(eval_examples, self.tokenizer, self.max_source_length,
                                                             self.max_target_length, stage='dev')
                all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
                all_source_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long)
                all_target_ids = torch.tensor([f.target_ids for f in eval_features], dtype=torch.long)
                all_target_mask = torch.tensor([f.target_mask for f in eval_features], dtype=torch.long)
                eval_data = TensorDataset(all_source_ids, all_source_mask, all_target_ids, all_target_mask)
                dev_dataset['dev_loss'] = eval_examples, eval_data
                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=eval_batch_size)

                logger.info("***** Running evaluation  *****")
                logger.info("  Num examples = %d", len(eval_examples))
                logger.info("  Batch size = %d", eval_batch_size)
                logger.info("  Num epoch = %d", cur_epoch)
                self.model.eval()
                eval_loss, batch_num = 0, 0
                for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
                    batch = tuple(t.to(self.device) for t in batch)
                    source_ids, source_mask, target_ids, target_mask = batch

                    with torch.no_grad():
                        outputs = self.model(input_ids=source_ids, attention_mask=source_mask,
                                                 labels=target_ids, decoder_attention_mask=target_mask)
                        loss = outputs.loss
                    eval_loss = eval_loss + loss.item()
                    batch_num += 1
                self.model.train()
                eval_loss = eval_loss / batch_num
                result = {'eval_ppl': round(numpy.exp(eval_loss), 5),
                          'global_step': global_step + 1,
                          'train_loss': round(train_loss, 5)}
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                logger.info("  " + "*" * 20)

                logger.info("***** CUDA.empty_cache() *****")
                torch.cuda.empty_cache()
                if do_eval_bleu:
                    self.model.eval()
                    df = pd.read_csv(eval_filename)
                    to_predict = df["src"].tolist()
                    ref_list = df["tgt"].tolist()
                    all_outputs = []
                    # Batching
                    for batch in tqdm(
                            [to_predict[i: i + eval_batch_size] for i in range(0, len(to_predict), eval_batch_size)],
                            desc="Generating outputs", ):
                        input = self.tokenizer.batch_encode_plus(
                            batch,
                            max_length=self.max_source_length,
                            padding="max_length",
                            return_tensors="pt",
                            truncation=True,
                        )
                        input_ids = input["input_ids"].to(self.device)
                        source_mask = input["attention_mask"].to(self.device)
                        outputs = self.model.generate(input_ids,
                                                      attention_mask=source_mask,
                                                      num_beams=self.beam_size,
                                                      max_length=self.max_target_length)
                        all_outputs.extend(outputs.cpu().numpy())
                    hyp_list = [
                        self.tokenizer.decode(
                            output_id, skip_special_tokens=True, clean_up_tokenization_spaces=False
                        )
                        for output_id in all_outputs
                    ]

                    assert len(ref_list) == len(hyp_list)
                    df = pd.DataFrame(hyp_list)
                    df.to_csv("hyp_temp.csv", index=False, header=None)
                    df = pd.DataFrame(ref_list)
                    df.to_csv("ref_temp.csv", index=False, header=None)

                    bleu4, acc = get_bleu_socre("ref_temp.csv", "hyp_temp.csv", task=task)

                    logger.info('dev set: bleu = %.2f\nacc = %.2f' % (bleu4, acc))
                    logger.info("  " + "*" * 20)
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    if bleu4 > best_bleu:
                        df = pd.DataFrame(hyp_list)
                        df.to_csv(output_dir+"preds.csv", index=False, header=None)
                        df = pd.DataFrame(ref_list)
                        df.to_csv(output_dir+"golds.csv", index=False, header=None)
                        count = 0
                        logger.info("  Best bleu:%s", bleu4)
                        logger.info("  " + "*" * 20)
                        best_bleu = bleu4
                        # Save best checkpoint for best bleu
                        output_dir_bleu = os.path.join(output_dir, 'checkpoint-best-bleu')
                        if not os.path.exists(output_dir_bleu):
                            os.makedirs(output_dir_bleu)
                        model_to_save = self.model.module if hasattr(self.model,
                                                                'module') else self.model  # Only save the model it-self
                        output_model_file = os.path.join(output_dir_bleu, "pytorch_model.bin")
                        torch.save(model_to_save.state_dict(), output_model_file)
                    else:
                        count += 1
                        if count == early_stop:
                            break
            logger.info("***** CUDA.empty_cache() *****")
            torch.cuda.empty_cache()

    def test(self, batch_size, filename, output_dir, task):
        logger.info("  " + "***** Testing *****")
        logger.info("  Batch size = %d", batch_size)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        df = pd.read_csv(filename)

        to_predict = df["src"].tolist()
        ref_list = df["tgt"].tolist()

        all_outputs = []
        # Batching
        for batch in tqdm(
                [to_predict[i: i + batch_size] for i in range(0, len(to_predict), batch_size)],
                desc="Generating outputs", ):
            input = self.tokenizer.batch_encode_plus(
                batch,
                max_length=self.max_source_length,
                padding="max_length",
                return_tensors="pt",
                truncation=True,
            )
            input_ids = input["input_ids"].to(self.device)
            source_mask = input["attention_mask"].to(self.device)
            outputs = self.model.generate(input_ids,
                                              attention_mask=source_mask,
                                              num_beams=self.beam_size,
                                              max_length=self.max_target_length)
            all_outputs.extend(outputs.cpu().numpy())

        hyp_list = [
            self.tokenizer.decode(
                output_id, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            for output_id in all_outputs
        ]

        assert len(ref_list) == len(hyp_list)
        df = pd.DataFrame(ref_list)
        df.to_csv(output_dir+"/gold.csv", index=False, header=None)
        df = pd.DataFrame(hyp_list)
        df.to_csv(output_dir + "/"+self.model_type+".csv", index=False, header=None)
        bleu4, acc = get_bleu_socre(output_dir+"/gold.csv", output_dir + "/"+self.model_type+".csv", task)
        logger.info('test set: bleu = %.2f\nacc = %.2f' % (bleu4, acc))
        logger.info("  " + "*" * 20)

    def predict(self, src):
        input = self.tokenizer.encode_plus(
            src,
            max_length=self.max_source_length,
            padding="max_length",
            return_tensors="pt",
            truncation=True,
        )
        input_ids = input["input_ids"].to(self.device)
        source_mask = input["attention_mask"].to(self.device)
        outputs = self.model.generate(input_ids,
                                      attention_mask=source_mask,
                                      num_beams=self.beam_size,
                                      max_length=self.max_target_length)
        hyp = self.tokenizer.decode(
            outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return hyp

class Encoder_Model():
    def __init__(self, model_type, model_name_or_path, load_model_path, beam_size, max_source_length, max_target_length, method_type='enc'):
        self.model_type = model_type
        self.config, self.model, self.tokenizer = build_or_load_gen_model(model_type, model_name_or_path,
                                                                          load_model_path, method_type, max_target_length, beam_size)
        self.method_type = method_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.beam_size, self.max_source_length, self.max_target_length = beam_size, max_source_length, max_target_length

    def train(self, train_filename, train_batch_size, learning_rate, num_train_epochs, early_stop, task,
              do_eval, eval_filename, eval_batch_size, output_dir, do_eval_bleu, AT=False):

        train_examples = read_examples(train_filename)
        train_features = convert_examples_to_features(train_examples, self.tokenizer, self.max_source_length,
                                                      self.max_target_length, stage='train')

        all_source_ids = torch.tensor([f.source_ids for f in train_features], dtype=torch.long)
        all_source_mask = torch.tensor([f.source_mask for f in train_features], dtype=torch.long)
        all_target_ids = torch.tensor([f.target_ids for f in train_features], dtype=torch.long)
        all_target_mask = torch.tensor([f.target_mask for f in train_features], dtype=torch.long)

        train_data = TensorDataset(all_source_ids, all_source_mask, all_target_ids, all_target_mask)

        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                      batch_size=train_batch_size)

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        t_total = len(train_dataloader) // num_train_epochs
        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=1e-8)
        num_train_optimization_steps = num_train_epochs * len(train_dataloader)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=int(t_total * 0.1),
                                                    num_training_steps=num_train_optimization_steps)

        # Start training
        train_example_num = len(train_data)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", train_example_num)
        logger.info("  Batch size = %d", train_batch_size)
        logger.info("  Batch num = %d", math.ceil(train_example_num / train_batch_size))
        logger.info("  Num epoch = %d", num_train_epochs)
        dev_dataset = {}
        global_step, best_bleu, best_loss = 0, -1, 1e6
        count = 0
        npgd = NPGD(self.model)
        K = 3
        for cur_epoch in range(int(num_train_epochs)):
            bar = tqdm(train_dataloader, total=len(train_dataloader), desc="Training")
            nb_tr_examples, nb_tr_steps, tr_loss = 0, 0, 0
            self.model.train()
            for step, batch in enumerate(bar):
                batch = tuple(t.to(self.device) for t in batch)
                source_ids, source_mask, target_ids, target_mask = batch
                loss, _, _ = self.model(source_ids=source_ids, source_mask=source_mask,
                                     target_ids=target_ids, target_mask=target_mask)

                tr_loss += loss.item()
                nb_tr_examples += source_ids.size(0)
                nb_tr_steps += 1
                loss.backward()

                if AT == True:
                    npgd.backup_grad()
                    for t in range(K):
                        npgd.attack(emb_name='emb', is_first_attack=(t == 0))
                        if t != K-1:
                            self.model.zero_grad()
                        else:
                            npgd.restore_grad()
                        loss_adv, _, _ = self.model(source_ids=source_ids, source_mask=source_mask,
                                     target_ids=target_ids, target_mask=target_mask)
                        loss_adv.backward()
                    npgd.restore(emb_name='embeddings')

                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                train_loss = round(tr_loss * 1 / (nb_tr_steps + 1), 4)
                bar.set_description("[{}] Train loss {}".format(cur_epoch, round(train_loss, 3)))

            if do_eval == True:
                # Eval model with dev dataset
                eval_examples = read_examples(eval_filename)
                eval_features = convert_examples_to_features(eval_examples, self.tokenizer, self.max_source_length,
                                                             self.max_target_length, stage='dev')
                all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
                all_source_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long)
                all_target_ids = torch.tensor([f.target_ids for f in eval_features], dtype=torch.long)
                all_target_mask = torch.tensor([f.target_mask for f in eval_features], dtype=torch.long)
                eval_data = TensorDataset(all_source_ids, all_source_mask, all_target_ids, all_target_mask)
                dev_dataset['dev_loss'] = eval_examples, eval_data
                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=eval_batch_size)

                logger.info("***** Running evaluation  *****")
                logger.info("  Num examples = %d", len(eval_examples))
                logger.info("  Batch size = %d", eval_batch_size)
                logger.info("  Num epoch = %d", cur_epoch)
                self.model.eval()
                eval_loss, batch_num = 0, 0
                for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
                    batch = tuple(t.to(self.device) for t in batch)
                    source_ids, source_mask, target_ids, target_mask = batch

                    with torch.no_grad():
                        _, loss, num = self.model(source_ids=source_ids, source_mask=source_mask,
                                                  target_ids=target_ids, target_mask=target_mask)

                    eval_loss += loss.sum().item()
                    batch_num += num.sum().item()
                self.model.train()
                eval_loss = eval_loss / batch_num
                result = {'eval_ppl': round(numpy.exp(eval_loss), 5),
                          'global_step': global_step + 1,
                          'train_loss': round(train_loss, 5)}
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                logger.info("  " + "*" * 20)

                logger.info("***** CUDA.empty_cache() *****")
                torch.cuda.empty_cache()
                if do_eval_bleu and cur_epoch >= 10:
                    self.model.eval()
                    p = []
                    for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
                        batch = tuple(t.to(self.device) for t in batch)
                        source_ids, source_mask, target_ids, target_mask = batch
                        with torch.no_grad():
                            preds = self.model(source_ids=source_ids, source_mask=source_mask)
                            for pred in preds:
                                t = pred[0].cpu().numpy()
                                t = list(t)
                                if 0 in t:
                                    t = t[:t.index(0)]
                                text = self.tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                                p.append(text)

                    ref_list = []
                    hyp_list = []

                    for hyp, gold in zip(p, eval_examples):
                        ref_list.append(gold.target)
                        hyp_list.append(hyp)

                    assert len(ref_list) == len(hyp_list)
                    df = pd.DataFrame(hyp_list)
                    df.to_csv("hyp_temp.csv", index=False, header=None)
                    df = pd.DataFrame(ref_list)
                    df.to_csv("ref_temp.csv", index=False, header=None)

                    bleu4, acc = get_bleu_socre("ref_temp.csv", "hyp_temp.csv", task=task)

                    logger.info('dev set: bleu = %.2f\nacc = %.2f' % (bleu4, acc))
                    logger.info("  " + "*" * 20)
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    if bleu4 > best_bleu:
                        df = pd.DataFrame(hyp_list)
                        df.to_csv(output_dir + "preds.csv", index=False, header=None)
                        df = pd.DataFrame(ref_list)
                        df.to_csv(output_dir + "golds.csv", index=False, header=None)
                        count = 0
                        logger.info("  Best bleu:%s", bleu4)
                        logger.info("  " + "*" * 20)
                        best_bleu = bleu4
                        # Save best checkpoint for best bleu
                        output_dir_bleu = os.path.join(output_dir, 'checkpoint-best-bleu')
                        if not os.path.exists(output_dir_bleu):
                            os.makedirs(output_dir_bleu)
                        model_to_save = self.model.module if hasattr(self.model,
                                                                     'module') else self.model  # Only save the model it-self
                        output_model_file = os.path.join(output_dir_bleu, "pytorch_model.bin")
                        torch.save(model_to_save.state_dict(), output_model_file)
                    else:
                        count += 1
                        if count == early_stop:
                            break
            logger.info("***** CUDA.empty_cache() *****")
            torch.cuda.empty_cache()

    def test(self, batch_size, filename, output_dir, task):
        logger.info("  " + "***** Testing *****")
        logger.info("  Batch size = %d", batch_size)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        eval_examples = read_examples(filename)
        eval_features = convert_examples_to_features(eval_examples, self.tokenizer, self.max_source_length,
                                                     self.max_target_length, stage='dev')
        all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
        all_source_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_source_ids, all_source_mask)
        # Calculate bleu
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=batch_size)
        self.model.eval()
        p = []
        for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
            batch = tuple(t.to(self.device) for t in batch)
            source_ids, source_mask = batch
            with torch.no_grad():
                preds = self.model(source_ids=source_ids, source_mask=source_mask)
                for pred in preds:
                    t = pred[0].cpu().numpy()
                    t = list(t)
                    if 0 in t:
                        t = t[:t.index(0)]
                    text = self.tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    p.append(text)

        ref_list = []
        hyp_list = []

        for hyp, gold in zip(p, eval_examples):
            ref_list.append(gold.target)
            hyp_list.append(hyp)

        assert len(ref_list) == len(hyp_list)
        df = pd.DataFrame(ref_list)
        df.to_csv(output_dir + "/gold.csv", index=False, header=None)
        df = pd.DataFrame(hyp_list)
        df.to_csv(output_dir + "/"+self.model_type+".csv", index=False, header=None)
        bleu4, acc = get_bleu_socre(output_dir + "/gold.csv", output_dir + "/"+self.model_type+".csv", task)
        logger.info('test set: bleu = %.2f\nacc = %.2f' % (bleu4, acc))
        logger.info("  " + "*" * 20)

    def predict(self, src):
        self.model.eval()
        input = self.tokenizer.encode_plus(
            src,
            max_length=self.max_source_length,
            padding="max_length",
            return_tensors="pt",
            truncation=True,
        )
        input_ids = input["input_ids"].to(self.device)
        source_mask = input["attention_mask"].to(self.device)
        with torch.no_grad():
            preds = self.model(source_ids=input_ids, source_mask=source_mask)
            for pred in preds:
                t = pred[0].cpu().numpy()
                t = list(t)
                if 0 in t:
                    t = t[:t.index(0)]
                text = self.tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return text

class Decoder_Model():
    def __init__(self, model_type, model_name_or_path, load_model_path, beam_size, max_source_length, max_target_length, block_size, method_type='dec'):
        self.model_type = model_type
        self.config, self.model, self.tokenizer = build_or_load_gen_model(model_type, model_name_or_path,
                                                                          load_model_path, method_type, max_target_length, beam_size)
        self.method_type = method_type
        self.block_size = block_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.beam_size, self.max_source_length, self.max_target_length = beam_size, max_source_length, max_target_length

    def train(self, train_filename, train_batch_size, learning_rate, num_train_epochs, early_stop, task,
              do_eval, eval_filename, eval_batch_size, output_dir, do_eval_bleu, AT=False):

        train_data = GPTDataset(train_filename, tokenizer=self.tokenizer, block_size=self.block_size, mode='train')

        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                      batch_size=train_batch_size)

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        t_total = len(train_dataloader) // num_train_epochs
        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=1e-8)
        num_train_optimization_steps = num_train_epochs * len(train_dataloader)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=int(t_total * 0.1),
                                                    num_training_steps=num_train_optimization_steps)

        # Start training
        train_example_num = len(train_data)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", train_example_num)
        logger.info("  Batch size = %d", train_batch_size)
        logger.info("  Batch num = %d", math.ceil(train_example_num / train_batch_size))
        logger.info("  Num epoch = %d", num_train_epochs)

        global_step, best_bleu, best_loss = 0, -1, 1e6
        count = 0
        npgd = NPGD(self.model)
        K = 3
        for cur_epoch in range(int(num_train_epochs)):
            bar = tqdm(train_dataloader, total=len(train_dataloader), desc="Training")
            nb_tr_examples, nb_tr_steps, tr_loss = 0, 0, 0
            self.model.train()
            for step, (batch, token_labels) in enumerate(bar):
                inputs = batch.to(self.device)
                attn_mask = torch.tensor(token_labels.clone().detach() != 0, dtype=torch.uint8, device=self.device)
                loss_mask = torch.tensor(token_labels.clone().detach() == 2, dtype=torch.uint8, device=self.device)
                outputs = self.model(inputs, attention_mask=attn_mask)
                logits = outputs[0]
                labels = inputs
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                flatten_shift_loss_mask = loss_mask[..., :-1].contiguous().view(-1)
                ids = torch.nonzero(flatten_shift_loss_mask).view(-1)
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1))[ids], shift_labels.view(-1)[ids])
                tr_loss += loss.item()
                nb_tr_steps += 1
                loss.backward()

                if AT == True:
                    npgd.backup_grad()
                    for t in range(K):
                        npgd.attack(emb_name='wte', is_first_attack=(t == 0))
                        if t != K-1:
                            self.model.zero_grad()
                        else:
                            npgd.restore_grad()
                        outputs = self.model(inputs, attention_mask=attn_mask)
                        logits = outputs[0]
                        labels = inputs
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = labels[..., 1:].contiguous()
                        flatten_shift_loss_mask = loss_mask[..., :-1].contiguous().view(-1)
                        ids = torch.nonzero(flatten_shift_loss_mask).view(-1)
                        loss_adv = loss_fct(shift_logits.view(-1, shift_logits.size(-1))[ids], shift_labels.view(-1)[ids])
                        loss_adv.backward()
                    npgd.restore(emb_name='wte')

                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                train_loss = round(tr_loss * 1 / (nb_tr_steps + 1), 4)
                bar.set_description("[{}] Train loss {}".format(cur_epoch, round(train_loss, 3)))

            if do_eval == True:
                # Eval model with dev dataset
                eval_data = GPTDataset(eval_filename, tokenizer=self.tokenizer, block_size=self.block_size, mode='valid')
                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=eval_batch_size)

                logger.info("***** Running evaluation  *****")
                logger.info("  Num examples = %d", eval_data.__len__())
                logger.info("  Batch size = %d", eval_batch_size)
                logger.info("  Num epoch = %d", cur_epoch)
                self.model.eval()
                eval_loss, batch_num = 0, 0
                for step, (batch, token_labels) in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):
                    inputs = batch.to(self.device)
                    attn_mask = torch.tensor(token_labels.clone().detach() != 0, dtype=torch.uint8, device=self.device)
                    loss_mask = torch.tensor(token_labels.clone().detach() == 2, dtype=torch.uint8, device=self.device)
                    with torch.no_grad():
                        outputs = self.model(inputs, attention_mask=attn_mask)
                        logits = outputs[0]
                        labels = inputs
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = labels[..., 1:].contiguous()
                        # Flatten the tokens
                        loss_fct = CrossEntropyLoss()
                        flatten_shift_loss_mask = loss_mask[..., :-1].contiguous().view(-1)
                        ids = torch.nonzero(flatten_shift_loss_mask).view(-1)
                        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1))[ids], shift_labels.view(-1)[ids])
                    eval_loss += loss.mean().item()
                    batch_num += 1
                self.model.train()
                eval_loss = eval_loss / batch_num
                result = {'eval_ppl': round(numpy.exp(eval_loss), 5),
                          'global_step': global_step + 1,
                          'train_loss': round(train_loss, 5)}
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                logger.info("  " + "*" * 20)

                logger.info("***** CUDA.empty_cache() *****")
                torch.cuda.empty_cache()
                if do_eval_bleu:
                    self.model.eval()
                    pred_ids = []
                    for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Eval bleu"):
                        source_ids = batch[0].to(self.device)
                        with torch.no_grad():
                            preds = self.model.generate(source_ids,
                                                   use_cache=True,
                                                   num_beams=self.beam_size,
                                                   max_length=self.block_size)
                            top_preds = list(preds[:, source_ids.size(1):].cpu().numpy())
                            pred_ids.extend(top_preds)

                    hyp_list = [self.tokenizer.decode(id, skip_special_tokens=True, clean_up_tokenization_spaces=False) for id in
                             pred_ids]

                    datas = pd.read_csv(eval_filename)
                    ref_list = datas['tgt'].tolist()


                    assert len(ref_list) == len(hyp_list)
                    df = pd.DataFrame(hyp_list)
                    df.to_csv("hyp_temp.csv", index=False, header=None)
                    df = pd.DataFrame(ref_list)
                    df.to_csv("ref_temp.csv", index=False, header=None)

                    bleu4, acc = get_bleu_socre("ref_temp.csv", "hyp_temp.csv", task=task)

                    logger.info('dev set: bleu = %.2f\nacc = %.2f' % (bleu4, acc))
                    logger.info("  " + "*" * 20)
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    if bleu4 > best_bleu:
                        df = pd.DataFrame(hyp_list)
                        df.to_csv(output_dir + "preds.csv", index=False, header=None)
                        df = pd.DataFrame(ref_list)
                        df.to_csv(output_dir + "golds.csv", index=False, header=None)
                        count = 0
                        logger.info("  Best bleu:%s", bleu4)
                        logger.info("  " + "*" * 20)
                        best_bleu = bleu4
                        # Save best checkpoint for best bleu
                        output_dir_bleu = os.path.join(output_dir, 'checkpoint-best-bleu')
                        if not os.path.exists(output_dir_bleu):
                            os.makedirs(output_dir_bleu)
                        model_to_save = self.model.module if hasattr(self.model,
                                                                     'module') else self.model  # Only save the model it-self
                        output_model_file = os.path.join(output_dir_bleu, "pytorch_model.bin")
                        torch.save(model_to_save.state_dict(), output_model_file)
                    else:
                        count += 1
                        if count == early_stop:
                            break
            logger.info("***** CUDA.empty_cache() *****")
            torch.cuda.empty_cache()

    def test(self, batch_size, filename, output_dir, task):
        logger.info("  " + "***** Testing *****")
        logger.info("  Batch size = %d", batch_size)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        eval_data = GPTDataset(filename, tokenizer=self.tokenizer, block_size=self.block_size, mode='test')
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=batch_size)
        self.model.eval()
        pred_ids = []
        for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Eval bleu"):
            source_ids = batch[0].to(self.device)
            with torch.no_grad():
                preds = self.model.generate(source_ids,
                                            use_cache=True,
                                            num_beams=self.beam_size,
                                            max_length=self.block_size)
                top_preds = list(preds[:, source_ids.size(1):].cpu().numpy())
                pred_ids.extend(top_preds)

        hyp_list = [self.tokenizer.decode(id, skip_special_tokens=True, clean_up_tokenization_spaces=False) for id in
                    pred_ids]

        datas = pd.read_csv(filename)
        ref_list = datas['tgt'].tolist()

        assert len(ref_list) == len(hyp_list)
        df = pd.DataFrame(ref_list)
        df.to_csv(output_dir + "/gold.csv", index=False, header=None)
        df = pd.DataFrame(hyp_list)
        df.to_csv(output_dir + "/"+self.model_type+".csv", index=False, header=None)
        bleu4, acc = get_bleu_socre(output_dir + "/gold.csv", output_dir + "/"+self.model_type+".csv", task)
        logger.info('test set: bleu = %.2f\nacc = %.2f' % (bleu4, acc))
        logger.info("  " + "*" * 20)

    def predict(self, src):
        src = self.tokenizer.encode(src)
        tgt = []
        while (len(src) + len(tgt) + 2 > self.block_size):
            if (len(tgt) > len(src)):
                tgt = tgt[:-1]
            else:
                src = src[:-1]
        inputs = src + [self.tokenizer.bos_token_id]
        source_ids = torch.tensor([inputs]).to(self.device)
        with torch.no_grad():
            preds = self.model.generate(source_ids,
                                        use_cache=True,
                                        num_beams=self.beam_size,
                                        max_length=self.block_size)
            top_preds = list(preds[:, source_ids.size(1):].cpu().numpy())

            text = self.tokenizer.decode(
                top_preds[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
        return text

class UniXcoder():
    def __init__(self, model_type, model_name_or_path, load_model_path, beam_size, max_source_length, max_target_length, method_type='unixcoder'):
        self.model_type = model_type
        self.config, self.model, self.tokenizer = build_or_load_gen_model(model_type, model_name_or_path,
                                                                          load_model_path, method_type, max_target_length, beam_size)
        self.method_type = method_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.beam_size, self.max_source_length, self.max_target_length = beam_size, max_source_length, max_target_length

    def train(self, train_filename, train_batch_size, learning_rate, num_train_epochs, early_stop, task,
              do_eval, eval_filename, eval_batch_size, output_dir, do_eval_bleu, AT=False):

        train_examples = read_examples(train_filename)
        train_features = convert_examples_to_features(train_examples, self.tokenizer, self.max_source_length,
                                                      self.max_target_length, stage='train')

        all_source_ids = torch.tensor([f.source_ids for f in train_features], dtype=torch.long)
        all_source_mask = torch.tensor([f.source_mask for f in train_features], dtype=torch.long)
        all_target_ids = torch.tensor([f.target_ids for f in train_features], dtype=torch.long)
        all_target_mask = torch.tensor([f.target_mask for f in train_features], dtype=torch.long)

        train_data = TensorDataset(all_source_ids, all_source_mask, all_target_ids, all_target_mask)

        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                      batch_size=train_batch_size)

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        t_total = len(train_dataloader) // num_train_epochs
        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=1e-8)
        num_train_optimization_steps = num_train_epochs * len(train_dataloader)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=int(t_total * 0.1),
                                                    num_training_steps=num_train_optimization_steps)

        # Start training
        train_example_num = len(train_data)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", train_example_num)
        logger.info("  Batch size = %d", train_batch_size)
        logger.info("  Batch num = %d", math.ceil(train_example_num / train_batch_size))
        logger.info("  Num epoch = %d", num_train_epochs)
        dev_dataset = {}
        global_step, best_bleu, best_loss = 0, -1, 1e6
        count = 0
        npgd = NPGD(self.model)
        K = 3
        for cur_epoch in range(int(num_train_epochs)):
            bar = tqdm(train_dataloader, total=len(train_dataloader), desc="Training")
            nb_tr_examples, nb_tr_steps, tr_loss = 0, 0, 0
            self.model.train()
            for step, batch in enumerate(bar):
                batch = tuple(t.to(self.device) for t in batch)
                source_ids, source_mask, target_ids, target_mask = batch
                loss, _, _ = self.model(source_ids=source_ids, target_ids=target_ids)

                tr_loss += loss.item()
                nb_tr_examples += source_ids.size(0)
                nb_tr_steps += 1
                loss.backward()

                if AT == True:
                    npgd.backup_grad()
                    for t in range(K):
                        npgd.attack(emb_name='emb', is_first_attack=(t == 0))
                        if t != K-1:
                            self.model.zero_grad()
                        else:
                            npgd.restore_grad()
                        loss_adv, _, _ = self.model(source_ids=source_ids, target_ids=target_ids)
                        loss_adv.backward()
                    npgd.restore(emb_name='emb')

                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                train_loss = round(tr_loss * 1 / (nb_tr_steps + 1), 4)
                bar.set_description("[{}] Train loss {}".format(cur_epoch, round(train_loss, 3)))

            if do_eval == True:
                # Eval model with dev dataset
                eval_examples = read_examples(eval_filename)
                eval_features = convert_examples_to_features(eval_examples, self.tokenizer, self.max_source_length,
                                                             self.max_target_length, stage='dev')
                all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
                all_source_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long)
                all_target_ids = torch.tensor([f.target_ids for f in eval_features], dtype=torch.long)
                all_target_mask = torch.tensor([f.target_mask for f in eval_features], dtype=torch.long)
                eval_data = TensorDataset(all_source_ids, all_source_mask, all_target_ids, all_target_mask)
                dev_dataset['dev_loss'] = eval_examples, eval_data
                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=eval_batch_size)

                logger.info("***** Running evaluation  *****")
                logger.info("  Num examples = %d", len(eval_examples))
                logger.info("  Batch size = %d", eval_batch_size)
                logger.info("  Num epoch = %d", cur_epoch)
                self.model.eval()
                eval_loss, batch_num = 0, 0
                for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
                    batch = tuple(t.to(self.device) for t in batch)
                    source_ids, source_mask, target_ids, target_mask = batch

                    with torch.no_grad():
                        _, loss, num = self.model(source_ids=source_ids, target_ids=target_ids)

                    eval_loss += loss.sum().item()
                    batch_num += num.sum().item()
                self.model.train()
                eval_loss = eval_loss / batch_num
                result = {'eval_ppl': round(numpy.exp(eval_loss), 5),
                          'global_step': global_step + 1,
                          'train_loss': round(train_loss, 5)}
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                logger.info("  " + "*" * 20)

                logger.info("***** CUDA.empty_cache() *****")
                torch.cuda.empty_cache()
                if do_eval_bleu:
                    self.model.eval()
                    p = []
                    for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
                        batch = tuple(t.to(self.device) for t in batch)
                        source_ids, source_mask, target_ids, target_mask = batch
                        with torch.no_grad():
                            preds = self.model(source_ids=source_ids)
                            for pred in preds:
                                t = pred[0].cpu().numpy()
                                t = list(t)
                                if 0 in t:
                                    t = t[:t.index(0)]
                                text = self.tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                                p.append(text)

                    ref_list = []
                    hyp_list = []

                    for hyp, gold in zip(p, eval_examples):
                        ref_list.append(gold.target)
                        hyp_list.append(hyp)

                    assert len(ref_list) == len(hyp_list)
                    df = pd.DataFrame(hyp_list)
                    df.to_csv("hyp_temp.csv", index=False, header=None)
                    df = pd.DataFrame(ref_list)
                    df.to_csv("ref_temp.csv", index=False, header=None)

                    bleu4, acc = get_bleu_socre("ref_temp.csv", "hyp_temp.csv", task=task)

                    logger.info('dev set: bleu = %.2f\nacc = %.2f' % (bleu4, acc))
                    logger.info("  " + "*" * 20)
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    if bleu4 > best_bleu:
                        df = pd.DataFrame(hyp_list)
                        df.to_csv(output_dir + "preds.csv", index=False, header=None)
                        df = pd.DataFrame(ref_list)
                        df.to_csv(output_dir + "golds.csv", index=False, header=None)
                        count = 0
                        logger.info("  Best bleu:%s", bleu4)
                        logger.info("  " + "*" * 20)
                        best_bleu = bleu4
                        # Save best checkpoint for best bleu
                        output_dir_bleu = os.path.join(output_dir, 'checkpoint-best-bleu')
                        if not os.path.exists(output_dir_bleu):
                            os.makedirs(output_dir_bleu)
                        model_to_save = self.model.module if hasattr(self.model,
                                                                     'module') else self.model  # Only save the model it-self
                        output_model_file = os.path.join(output_dir_bleu, "pytorch_model.bin")
                        torch.save(model_to_save.state_dict(), output_model_file)
                    else:
                        count += 1
                        if count == early_stop:
                            break
            logger.info("***** CUDA.empty_cache() *****")
            torch.cuda.empty_cache()

    def test(self, batch_size, filename, output_dir, task):
        logger.info("  " + "***** Testing *****")
        logger.info("  Batch size = %d", batch_size)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        eval_examples = read_examples(filename)
        eval_features = convert_examples_to_features(eval_examples, self.tokenizer, self.max_source_length,
                                                     self.max_target_length, stage='dev')
        all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
        all_source_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_source_ids, all_source_mask)
        # Calculate bleu
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=batch_size)
        self.model.eval()
        p = []
        for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
            batch = tuple(t.to(self.device) for t in batch)
            source_ids, source_mask = batch
            with torch.no_grad():
                preds = self.model(source_ids=source_ids)
                for pred in preds:
                    t = pred[0].cpu().numpy()
                    t = list(t)
                    if 0 in t:
                        t = t[:t.index(0)]
                    text = self.tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    p.append(text)

        ref_list = []
        hyp_list = []

        for hyp, gold in zip(p, eval_examples):
            ref_list.append(gold.target)
            hyp_list.append(hyp)

        assert len(ref_list) == len(hyp_list)
        df = pd.DataFrame(ref_list)
        df.to_csv(output_dir + "/gold.csv", index=False, header=None)
        df = pd.DataFrame(hyp_list)
        df.to_csv(output_dir + "/"+self.model_type+".csv", index=False, header=None)
        bleu4, acc = get_bleu_socre(output_dir + "/gold.csv", output_dir + "/"+self.model_type+".csv", task)
        logger.info('test set: bleu = %.2f\nacc = %.2f' % (bleu4, acc))
        logger.info("  " + "*" * 20)

    def predict(self, src):
        input = self.tokenizer.encode_plus(
            src,
            max_length=self.max_source_length,
            padding="max_length",
            return_tensors="pt",
            truncation=True,
        )
        input_ids = input["input_ids"].to(self.device)

        with torch.no_grad():
            preds = self.model(source_ids=input_ids)
            for pred in preds:
                t = pred[0].cpu().numpy()
                t = list(t)
                if 0 in t:
                    t = t[:t.index(0)]
                text = self.tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return text

