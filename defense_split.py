import re
import numpy as np
import pandas as pd
import random
import torch
from transformers import RobertaModel, RobertaTokenizer
from tqdm import tqdm

from PLM import Encoder_Decoder, UniXcoder, Decoder_Model, Encoder_Model
from codegen.preprocessing.lang_processors.java_processor import JavaProcessor
from codegen.preprocessing.lang_processors.python_processor import PythonProcessor
from compile import check_java_code, check_java_code_testcases, check_python_code, check_python_code_testcases
from program_transformer.transformations import BlockSwap, ForWhileTransformer, OperandSwap, ConfusionRemover
from program_transformer.transformations.syntactic_noising_transformation import SyntacticNoisingTransformation
from utils import set_seed

pyprocessor = PythonProcessor()
jprocessor = JavaProcessor("./third_party")
tokenizer = RobertaTokenizer.from_pretrained('/home/yangguang/models/codebert-base')
codebert_model = RobertaModel.from_pretrained('/home/yangguang/models/codebert-base')
codebert_model = codebert_model.cuda()

def code_to_vecs(code, tokenizer, model):
    with torch.no_grad():
        inputs = tokenizer(code, return_tensors='pt', padding=True, truncation=True, max_length=256)
        inputs['input_ids'] = inputs['input_ids'].cuda()
        inputs['attention_mask'] = inputs['attention_mask'].cuda()
        hidden_size = model(**inputs, return_dict=True, output_hidden_states=True).hidden_states
        output = (hidden_size[-1] + hidden_size[1]).mean(dim=1)
    return output.cpu().numpy()[0]

def sim_score(code_A, code_B):
    vec_A = code_to_vecs(code_A, tokenizer, codebert_model)
    vec_B = code_to_vecs(code_B, tokenizer, codebert_model)
    cos = np.dot(vec_A, vec_B) / (np.linalg.norm(vec_A) * np.linalg.norm(vec_B))
    return cos

def decode(code, lang):
    if lang == 'python':
        code = pyprocessor.detokenize_code(code)
        codes = code.split('\n')
        code = '\n'.join([s for s in codes if len(s.strip()) > 0])
    else:
        code = jprocessor.detokenize_code(code)
        codes = code.split('\n')
        code = '\n'.join([s for s in codes if len(s.strip()) > 0])
    code = code.replace('\n', ' ')
    return ' '.join(code.split())

def encode(code, lang):
    if lang == 'python':
        return ' '.join(pyprocessor.tokenize_code(code))
    else:
        return ' '.join(jprocessor.tokenize_code(code))

def transform_java_code(java, operand_swap, operand_transform, if_else_transform, for_while_transform, noisy_transform):
    # print('开始转换...')
    # 进行语法转换
    code, meta = '', ''
    decode_code = decode(java.replace("Translate Java to Python: ", ""), 'java')
    temp_code_dict = {}
    operand_swap_code, operand_swap_meta = operand_swap.transform_code(decode_code)
    operand_transform_code, operand_transform_meta = operand_transform.transform_code(decode_code)
    if_else_code, if_else_meta = if_else_transform.transform_code(decode_code)
    for_while_code, for_while_meta = for_while_transform.transform_code(decode_code)
    if operand_swap_meta['success']==True :
        # operand_swap操作成功
        temp_code_dict['operand_swap'] = operand_swap_code
        operand_swap_transform_code, operand_swap_transform_meta = operand_transform.transform_code(operand_swap_code)
        operand_swap_if_else_code, operand_swap_if_else_meta = if_else_transform.transform_code(operand_swap_code)
        operand_swap_for_while_code, operand_swap_for_while_meta = for_while_transform.transform_code(operand_swap_code)
        if operand_swap_transform_meta['success'] == True:
            temp_code_dict['operand_swap_transform'] = operand_swap_transform_code
            operand_swap_transform_if_else_code, operand_swap_transform_if_else_meta = if_else_transform.transform_code(
                operand_swap_transform_code)
            operand_swap_transform_for_while_code, operand_swap_transform_for_while_meta = for_while_transform.transform_code(
                operand_swap_transform_code)
            if operand_swap_transform_if_else_meta['success'] == True:
                temp_code_dict['operand_swap_transform_if_else'] = operand_swap_transform_if_else_code
                operand_swap_transform_if_else_for_while_code, operand_swap_transform_if_else_for_while_meta = for_while_transform.transform_code(
                    operand_swap_transform_if_else_code)
                if operand_swap_transform_if_else_for_while_meta['success'] == True:
                    temp_code_dict['operand_swap_transform_if_else_for_while'] = operand_swap_transform_if_else_for_while_code
            if operand_swap_transform_for_while_meta['success'] == True:
                temp_code_dict['operand_swap_for_while'] = operand_swap_transform_for_while_code
        if operand_swap_if_else_meta['success'] == True:
            temp_code_dict['operand_swap_if_else'] = operand_swap_if_else_code
            operand_swap_if_else_for_while_code, operand_swap_if_else_for_while_meta = for_while_transform.transform_code(
                operand_swap_if_else_code)
            if operand_swap_if_else_for_while_meta['success'] == True:
                temp_code_dict['operand_swap_if_else_for_while'] = operand_swap_if_else_for_while_code
        if operand_swap_for_while_meta['success'] == True:
            temp_code_dict['operand_swap_for_while'] = operand_swap_for_while_code

    if operand_transform_meta['success']==True:
        temp_code_dict['operand_transform'] = operand_transform_code
        operand_transform_if_else_code, operand_transform_if_else_meta = if_else_transform.transform_code(operand_transform_code)
        operand_transform_for_while_code, operand_transform_for_while_meta = for_while_transform.transform_code(operand_transform_code)
        if operand_transform_if_else_meta['success'] == True:
            temp_code_dict['operand_transform_if_else'] = operand_transform_if_else_code
            operand_transform_if_else_for_while_code, operand_transform_if_else_for_while_meta = for_while_transform.transform_code(
                operand_transform_if_else_code)
            if operand_transform_if_else_for_while_meta['success'] == True:
                temp_code_dict['operand_transform_if_else_for_while'] = operand_transform_if_else_for_while_code
        if operand_transform_for_while_meta['success'] == True:
            temp_code_dict['operand_transform_for_while'] = operand_transform_for_while_code

    if if_else_meta['success']==True:
        # if-esle操作成功
        temp_code_dict['if_else'] = if_else_code
        if_else_for_while_code, if_else_for_while_meta = for_while_transform.transform_code(if_else_code)
        if if_else_for_while_meta['success'] == True:
            temp_code_dict['if_else_for_while'] = if_else_for_while_code

    if for_while_meta['success']==True:
        # for-while操作成功
        temp_code_dict['for_while'] = for_while_code


    # 如果语法转换成功的个数为0，则使用noisy-transformation
    # if len(temp_code_dict.keys()) == 0:
    #     noisy_code, noisy_meta = noisy_transform.transform_code(decode_code)
    #     if noisy_meta['success'] == True:
    #         temp_code_dict['noisy'] = noisy_code

    return temp_code_dict

def transform_python_code(python, operand_swap, operand_transform, if_else_transform, for_while_transform, noisy_transform):
    # print('开始转换...')
    code, meta = '', ''
    decode_code = decode(python.replace("Translate Python to Java: ", ""), 'python')
    temp_code_dict = {}
    operand_swap_code, operand_swap_meta = operand_swap.transform_code(decode_code)
    operand_transform_code, operand_transform_meta = operand_transform.transform_code(decode_code)
    if_else_code, if_else_meta = if_else_transform.transform_code(decode_code)
    for_while_code, for_while_meta = for_while_transform.transform_code(decode_code)
    if operand_swap_meta['success'] == True:
        # operand_swap操作成功
        temp_code_dict['operand_swap'] = operand_swap_code
        operand_swap_transform_code, operand_swap_transform_meta = operand_transform.transform_code(operand_swap_code)
        operand_swap_if_else_code, operand_swap_if_else_meta = if_else_transform.transform_code(operand_swap_code)
        operand_swap_for_while_code, operand_swap_for_while_meta = for_while_transform.transform_code(operand_swap_code)
        if operand_swap_transform_meta['success'] == True:
            temp_code_dict['operand_swap_transform'] = operand_swap_transform_code
            operand_swap_transform_if_else_code, operand_swap_transform_if_else_meta = if_else_transform.transform_code(
                operand_swap_transform_code)
            operand_swap_transform_for_while_code, operand_swap_transform_for_while_meta = for_while_transform.transform_code(
                operand_swap_transform_code)
            if operand_swap_transform_if_else_meta['success'] == True:
                temp_code_dict['operand_swap_transform_if_else'] = operand_swap_transform_if_else_code
                operand_swap_transform_if_else_for_while_code, operand_swap_transform_if_else_for_while_meta = for_while_transform.transform_code(
                    operand_swap_transform_if_else_code)
                if operand_swap_transform_if_else_for_while_meta['success'] == True:
                    temp_code_dict[
                        'operand_swap_transform_if_else_for_while'] = operand_swap_transform_if_else_for_while_code
            if operand_swap_transform_for_while_meta['success'] == True:
                temp_code_dict['operand_swap_for_while'] = operand_swap_transform_for_while_code
        if operand_swap_if_else_meta['success'] == True:
            temp_code_dict['operand_swap_if_else'] = operand_swap_if_else_code
            operand_swap_if_else_for_while_code, operand_swap_if_else_for_while_meta = for_while_transform.transform_code(
                operand_swap_if_else_code)
            if operand_swap_if_else_for_while_meta['success'] == True:
                temp_code_dict['operand_swap_if_else_for_while'] = operand_swap_if_else_for_while_code
        if operand_swap_for_while_meta['success'] == True:
            temp_code_dict['operand_swap_for_while'] = operand_swap_for_while_code

    if operand_transform_meta['success'] == True:
        temp_code_dict['operand_transform'] = operand_transform_code
        operand_transform_if_else_code, operand_transform_if_else_meta = if_else_transform.transform_code(
            operand_transform_code)
        operand_transform_for_while_code, operand_transform_for_while_meta = for_while_transform.transform_code(
            operand_transform_code)
        if operand_transform_if_else_meta['success'] == True:
            temp_code_dict['operand_transform_if_else'] = operand_transform_if_else_code
            operand_transform_if_else_for_while_code, operand_transform_if_else_for_while_meta = for_while_transform.transform_code(
                operand_transform_if_else_code)
            if operand_transform_if_else_for_while_meta['success'] == True:
                temp_code_dict['operand_transform_if_else_for_while'] = operand_transform_if_else_for_while_code
        if operand_transform_for_while_meta['success'] == True:
            temp_code_dict['operand_transform_for_while'] = operand_transform_for_while_code

    if if_else_meta['success'] == True:
        # if-esle操作成功
        temp_code_dict['if_else'] = if_else_code
        if_else_for_while_code, if_else_for_while_meta = for_while_transform.transform_code(if_else_code)
        if if_else_for_while_meta['success'] == True:
            temp_code_dict['if_else_for_while'] = if_else_for_while_code

    if for_while_meta['success'] == True:
        # for-while操作成功
        temp_code_dict['for_while'] = for_while_code

    # 如果语法转换成功的个数为0，则使用noisy-transformation
    # if len(temp_code_dict.keys()) == 0:
    #     noisy_code, noisy_meta = noisy_transform.transform_code(decode_code)
    #     temp_code_dict['noisy'] = noisy_code

    return temp_code_dict

def sampling(java, java_dict, python, python_dict):
    java_keys = list(java_dict.keys())
    python_keys = list(python_dict.keys())
    jiaoji = [x for x in java_keys if x in python_keys]

    java = decode(java, 'java')
    python = decode(python, 'python')

    temp_scores = {}

    for k in jiaoji:
        score_java = sim_score(java, decode(java_dict[k], 'java'))
        score_python = sim_score(python, decode(python_dict[k], 'python'))
        temp_scores[k] = score_java + score_python
    # print(temp_scores)
    key = min(temp_scores, key=lambda x:temp_scores[x])
    return key

def generate_dataset_defense(random_seed):
    random.seed(random_seed)
    # a>b --> b<a
    java_operand_swap = OperandSwap(
        '/home/yangguang/PycharmProjects/RobustCodeTrans/evaluation/CodeBLEU/parser/my-languages.so', 'java'
    )
    # a+=b --> a=a+b
    java_operand_transform = ConfusionRemover(
        '/home/yangguang/PycharmProjects/RobustCodeTrans/evaluation/CodeBLEU/parser/my-languages.so', 'java'
    )
    java_if_else_transform = BlockSwap(
        '/home/yangguang/PycharmProjects/RobustCodeTrans/evaluation/CodeBLEU/parser/my-languages.so', 'java'
    )
    java_for_while_transform = ForWhileTransformer(
        '/home/yangguang/PycharmProjects/RobustCodeTrans/evaluation/CodeBLEU/parser/my-languages.so', 'java'
    )
    java_noisy_transform = SyntacticNoisingTransformation(
        '/home/yangguang/PycharmProjects/RobustCodeTrans/evaluation/CodeBLEU/parser/my-languages.so', 'java'
    )

    # a>b --> b<a
    python_operand_swap = OperandSwap(
        '/home/yangguang/PycharmProjects/RobustCodeTrans/evaluation/CodeBLEU/parser/my-languages.so', 'python'
    )
    # a+=b --> a=a+b
    python_operand_transform = ConfusionRemover(
        '/home/yangguang/PycharmProjects/RobustCodeTrans/evaluation/CodeBLEU/parser/my-languages.so', 'python'
    )
    python_if_else_transform = BlockSwap(
        '/home/yangguang/PycharmProjects/RobustCodeTrans/evaluation/CodeBLEU/parser/my-languages.so', 'python'
    )
    python_for_while_transform = ForWhileTransformer(
        '/home/yangguang/PycharmProjects/RobustCodeTrans/evaluation/CodeBLEU/parser/my-languages.so', 'python'
    )
    python_noisy_transform = SyntacticNoisingTransformation(
        '/home/yangguang/PycharmProjects/RobustCodeTrans/evaluation/CodeBLEU/parser/my-languages.so', 'python'
    )

    df = pd.read_csv('dataset/train_j2p.csv')
    tgts = df['tgt'].tolist()
    srcs = df['src'].tolist()
    j2p_datas = []
    p2j_datas = []
    count = 0
    for i in tqdm(range(len(srcs))):
        # 将src代码进行语法转换
        java_dict = transform_java_code(srcs[i], java_operand_swap, java_operand_transform, java_if_else_transform,
                                        java_for_while_transform, java_noisy_transform)

        python_dict = transform_python_code(tgts[i], python_operand_swap, python_operand_transform, python_if_else_transform,
                                        python_for_while_transform, python_noisy_transform)

        java_keys = list(java_dict.keys())
        python_keys = list(python_dict.keys())

        jiaoji = [x for x in java_keys if x in python_keys]

        if len(jiaoji)==0:
            key = 'non'
        else:
            key = "sampling"
        #     key = random.choice(jiaoji)

        if key == 'non':
            # try:
            #     j2p_datas.append(['Translate Java to Python: ' + java_dict[java_keys[0]], tgts[i]])
            # except:
            #     continue
            # try:
            #     p2j_datas.append(
            #         ['Translate Python to Java: ' + python_dict[python_keys[0]], srcs[i].replace('Translate Java to Python: ', '')])
            # except:
            #     continue
            continue
        else:
            key = sampling(srcs[i].replace('Translate Java to Python: ',''), java_dict, tgts[i], python_dict)
            count += 1
            j2p_datas.append([decode(java_dict[key], 'java')])
            p2j_datas.append([decode(python_dict[key], 'python')])
    print(count)
    df = pd.DataFrame(j2p_datas, columns=['java'])
    df.to_csv("dataset_defense/java_aug.csv", index=False)
    df = pd.DataFrame(p2j_datas, columns=['python'])
    df.to_csv("dataset_defense/python_aug.csv", index=False)

if __name__ == '__main__':
    # for seed in ['1234', '42', '1', '1024', '256']:
    random.seed('1234')
    generate_dataset_defense('1234')
