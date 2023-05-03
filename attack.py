import re

import pandas as pd
import random

from tqdm import tqdm

from PLM import Encoder_Decoder, UniXcoder, Decoder_Model, Encoder_Model
from codegen.preprocessing.lang_processors.java_processor import JavaProcessor
from codegen.preprocessing.lang_processors.python_processor import PythonProcessor
from compile import check_java_code, check_java_code_testcases, check_python_code, check_python_code_testcases
from program_transformer.transformations import BlockSwap, ForWhileTransformer, OperandSwap, ConfusionRemover
from utils import set_seed

pyprocessor = PythonProcessor()
jprocessor = JavaProcessor("./third_party")

def decode(code, lang):
    if lang == 'python':
        return pyprocessor.detokenize_code(code)
    else:
        return jprocessor.detokenize_code(code)

def encode(code, lang):
    if lang == 'python':
        return ' '.join(pyprocessor.tokenize_code(code))
    else:
        return ' '.join(jprocessor.tokenize_code(code))


def transform_java_code(model, java, i, operand_swap, operand_transform, if_else_transform, for_while_transform):
    # print('开始转换...')
    # 先检查为攻击前model生成的java代码是否能够通过测试用例，如果能通过，则直接跳过
    code = model.predict(java)
    program = pyprocessor.detokenize_code(code)
    try:
        state = check_python_code_testcases(program, 'test_cases_templates/' + str(i) + '.txt')
    except:
        state = False
    if state == False:
        code, meta = "", "fail"
        # print('原始生成的代码就不能通过测试用例，直接返回')
        return code, meta
    # 如果不能通过，则进行语法转换
    code, meta = '', ''
    decode_code = decode(java.replace("Translate Java to Python: ", ""), 'java')
    temp_code_list = []
    operand_swap_code, operand_swap_meta = operand_swap.transform_code(decode_code)
    operand_transform_code, operand_transform_meta = operand_transform.transform_code(decode_code)
    if_else_code, if_else_meta = if_else_transform.transform_code(decode_code)
    for_while_code, for_while_meta = for_while_transform.transform_code(decode_code)
    if operand_swap_meta['success']==True :
        # operand_swap操作成功
        temp_code_list.append(operand_swap_code)
        operand_swap_transform_code, operand_swap_transform_meta = operand_transform.transform_code(operand_swap_code)
        operand_swap_if_else_code, operand_swap_if_else_meta = if_else_transform.transform_code(operand_swap_code)
        operand_swap_for_while_code, operand_swap_for_while_meta = for_while_transform.transform_code(operand_swap_code)
        if operand_swap_transform_meta['success'] == True:
            temp_code_list.append(operand_swap_transform_code)
            operand_swap_transform_if_else_code, operand_swap_transform_if_else_meta = if_else_transform.transform_code(
                operand_swap_transform_code)
            operand_swap_transform_for_while_code, operand_swap_transform_for_while_meta = for_while_transform.transform_code(
                operand_swap_transform_code)
            if operand_swap_transform_if_else_meta['success'] == True:
                temp_code_list.append(operand_swap_transform_if_else_code)
                operand_swap_transform_if_else_for_while_code, operand_swap_transform_if_else_for_while_meta = for_while_transform.transform_code(
                    operand_swap_transform_if_else_code)
                if operand_swap_transform_if_else_for_while_meta['success'] == True:
                    temp_code_list.append(operand_swap_transform_if_else_for_while_code)
            if operand_swap_transform_for_while_meta['success'] == True:
                temp_code_list.append(operand_swap_transform_for_while_code)
        if operand_swap_if_else_meta['success'] == True:
            temp_code_list.append(operand_swap_if_else_code)
            operand_swap_if_else_for_while_code, operand_swap_if_else_for_while_meta = for_while_transform.transform_code(
                operand_swap_if_else_code)
            if operand_swap_if_else_for_while_meta['success'] == True:
                temp_code_list.append(operand_swap_if_else_for_while_code)
        if operand_swap_for_while_meta['success'] == True:
            temp_code_list.append(operand_swap_for_while_code)

    if operand_transform_meta['success']==True:
        temp_code_list.append(operand_transform_code)
        operand_transform_if_else_code, operand_transform_if_else_meta = if_else_transform.transform_code(operand_transform_code)
        operand_transform_for_while_code, operand_transform_for_while_meta = for_while_transform.transform_code(operand_transform_code)
        if operand_transform_if_else_meta['success'] == True:
            temp_code_list.append(operand_transform_if_else_code)
            operand_transform_if_else_for_while_code, operand_transform_if_else_for_while_meta = for_while_transform.transform_code(
                operand_transform_if_else_code)
            if operand_transform_if_else_for_while_meta['success'] == True:
                temp_code_list.append(operand_transform_if_else_for_while_code)
        if operand_transform_for_while_meta['success'] == True:
            temp_code_list.append(operand_transform_for_while_code)

    if if_else_meta['success']==True:
        # if-esle操作成功
        temp_code_list.append(if_else_code)
        if_else_for_while_code, if_else_for_while_meta = for_while_transform.transform_code(if_else_code)
        if if_else_for_while_meta['success'] == True:
            temp_code_list.append(if_else_for_while_code)

    if for_while_meta['success']==True:
        # for-while操作成功
        temp_code_list.append(for_while_code)

    # 如果语法转换成功的个数为0，则直接返回
    if len(temp_code_list) == 0:
        code, meta = "", "fail"
        # print('语法转换成功的个数为0，直接返回')
        return code, meta
    # 去重
    temp_code_list = list(set(temp_code_list))
    # print('语法转换成功的个数: ', str(len(temp_code_list)))

    # 对转换成功的代码进行模型生成
    temp_python_code_list = [model.predict("Translate Java to Python: " + encode(program, 'java')) for program in temp_code_list]
    # 通过遍历进行攻击，找出能够破坏语义的代码
    for index in range(len(temp_python_code_list)):
        program = pyprocessor.detokenize_code(temp_python_code_list[index])
        state = check_python_code_testcases(program, 'test_cases_templates/' + str(i) + '.txt')
        # 如果破坏语义，则返回对抗样本
        if (state == False):
            code, meta = "Translate Java to Python: " + encode(temp_code_list[index], 'java'), "success"
            # print('攻击成功了，直接返回')
            return code, meta
        else:
            # print('攻击失败...继续攻击')
            code, meta = "", "fail"
    return code, meta


def transform_python_code(model, python, i, operand_swap, operand_transform, if_else_transform, for_while_transform):
    print('开始转换...')
    # 先检查为攻击前model生成的java代码是否能够通过测试用例，如果能通过，则直接跳过
    code = model.predict(python)
    program = jprocessor.detokenize_code(code)
    try:
        state = check_java_code_testcases(program, 'Main_'+str(i)+'.txt')
    except:
        state = False
    if state != True:
        code, meta = "", "fail"
        print('原始生成的代码就不能通过测试用例，直接返回')
        return code, meta
    # 如果不能通过，则进行语法转换
    code, meta = '', ''
    decode_code = decode(python.replace("Translate Python to Java: ", ""), 'python')
    temp_code_list = []
    operand_swap_code, operand_swap_meta = operand_swap.transform_code(decode_code)
    operand_transform_code, operand_transform_meta = operand_transform.transform_code(decode_code)
    if_else_code, if_else_meta = if_else_transform.transform_code(decode_code)
    for_while_code, for_while_meta = for_while_transform.transform_code(decode_code)
    if operand_swap_meta['success'] == True:
        # operand_swap操作成功
        temp_code_list.append(operand_swap_code)
        operand_swap_transform_code, operand_swap_transform_meta = operand_transform.transform_code(operand_swap_code)
        operand_swap_if_else_code, operand_swap_if_else_meta = if_else_transform.transform_code(operand_swap_code)
        operand_swap_for_while_code, operand_swap_for_while_meta = for_while_transform.transform_code(operand_swap_code)
        if operand_swap_transform_meta['success'] == True:
            temp_code_list.append(operand_swap_transform_code)
            operand_swap_transform_if_else_code, operand_swap_transform_if_else_meta = if_else_transform.transform_code(
                operand_swap_transform_code)
            operand_swap_transform_for_while_code, operand_swap_transform_for_while_meta = for_while_transform.transform_code(
                operand_swap_transform_code)
            if operand_swap_transform_if_else_meta['success'] == True:
                temp_code_list.append(operand_swap_transform_if_else_code)
                operand_swap_transform_if_else_for_while_code, operand_swap_transform_if_else_for_while_meta = for_while_transform.transform_code(
                    operand_swap_transform_if_else_code)
                if operand_swap_transform_if_else_for_while_meta['success'] == True:
                    temp_code_list.append(operand_swap_transform_if_else_for_while_code)
            if operand_swap_transform_for_while_meta['success'] == True:
                temp_code_list.append(operand_swap_transform_for_while_code)
        if operand_swap_if_else_meta['success'] == True:
            temp_code_list.append(operand_swap_if_else_code)
            operand_swap_if_else_for_while_code, operand_swap_if_else_for_while_meta = for_while_transform.transform_code(
                operand_swap_if_else_code)
            if operand_swap_if_else_for_while_meta['success'] == True:
                temp_code_list.append(operand_swap_if_else_for_while_code)
        if operand_swap_for_while_meta['success'] == True:
            temp_code_list.append(operand_swap_for_while_code)

    if operand_transform_meta['success'] == True:
        temp_code_list.append(operand_transform_code)
        operand_transform_if_else_code, operand_transform_if_else_meta = if_else_transform.transform_code(
            operand_transform_code)
        operand_transform_for_while_code, operand_transform_for_while_meta = for_while_transform.transform_code(
            operand_transform_code)
        if operand_transform_if_else_meta['success'] == True:
            temp_code_list.append(operand_transform_if_else_code)
            operand_transform_if_else_for_while_code, operand_transform_if_else_for_while_meta = for_while_transform.transform_code(
                operand_transform_if_else_code)
            if operand_transform_if_else_for_while_meta['success'] == True:
                temp_code_list.append(operand_transform_if_else_for_while_code)
        if operand_transform_for_while_meta['success'] == True:
            temp_code_list.append(operand_transform_for_while_code)

    if if_else_meta['success'] == True:
        # if-esle操作成功
        temp_code_list.append(if_else_code)
        if_else_for_while_code, if_else_for_while_meta = for_while_transform.transform_code(if_else_code)
        if if_else_for_while_meta['success'] == True:
            temp_code_list.append(if_else_for_while_code)

    if for_while_meta['success'] == True:
        # for-while操作成功
        temp_code_list.append(for_while_code)

    # 如果语法转换成功的个数为0，则直接返回
    if len(temp_code_list) == 0:
        code, meta = "", "fail"
        print('语法转换成功的个数为0，直接返回')
        return code, meta
    # 去重
    temp_code_list = list(set(temp_code_list))
    print('语法转换成功的个数: ', str(len(temp_code_list)))

    # 对转换成功的代码进行模型生成
    temp_python_code_list = [model.predict("Translate Python to Java: " + encode(program, 'python')) for program in
                             temp_code_list]
    # 通过遍历进行攻击，找出能够破坏语义的代码
    for index in range(len(temp_python_code_list)):
        program = jprocessor.detokenize_code(temp_python_code_list[index])
        state = check_java_code_testcases(program, 'Main_' + str(i) + '.txt')
        # 如果破坏语义，则返回对抗样本
        if (state != True):
            code, meta = "Translate Python to Java: " + encode(temp_code_list[index], 'python'), "success"
            print('攻击成功了，直接返回')
            return code, meta
        else:
            print('攻击失败...继续攻击')
            code, meta = "", "fail"
    return code, meta

def generate_dataset_attack(model_dict, model_type, task):
    if model_type == 'codet5' or model_type == 'plbart' or model_type == 'natgen':
        model = Encoder_Decoder(model_type=model_type, model_name_or_path=model_dict[model_type], beam_size=10,
                      max_source_length=350, max_target_length=350,
                      load_model_path='models/original/valid_output_'+task+'/'+model_type+'/checkpoint-best-bleu/pytorch_model.bin')
    if model_type == 'unixcoder':
        model = UniXcoder(model_type=model_type, model_name_or_path=model_dict[model_type], beam_size=10,
                          max_source_length=350, max_target_length=350,
                          load_model_path='models/original/valid_output_' + task + '/' + model_type + '/checkpoint-best-bleu/pytorch_model.bin')
    if 'codegpt' in model_type or 'codegen' in model_type:
        model = Decoder_Model(model_type=model_type, model_name_or_path=model_dict[model_type], beam_size=10,
                              block_size=700,
                              max_source_length=350, max_target_length=700,
                              load_model_path='models/original/valid_output_' + task + '/' + model_type + '/checkpoint-best-bleu/pytorch_model.bin')
    if 'codebert' in model_type or 'contrabert' in model_type:
        model = Encoder_Model(model_type=model_type, model_name_or_path=model_dict[model_type], beam_size=10,
                              max_source_length=350, max_target_length=350,
                              load_model_path='models/original/valid_output_' + task + '/' + model_type + '/checkpoint-best-bleu/pytorch_model.bin')
    count = 0
    if task == 'j2p':
        # a>b --> b<a
        operand_swap = OperandSwap(
            '/home/yangguang/PycharmProjects/RobustCodeTrans/evaluation/CodeBLEU/parser/my-languages.so', 'java'
        )
        # a+=b --> a=a+b
        operand_transform = ConfusionRemover(
            '/home/yangguang/PycharmProjects/RobustCodeTrans/evaluation/CodeBLEU/parser/my-languages.so', 'java'
        )
        if_else_transform = BlockSwap(
            '/home/yangguang/PycharmProjects/RobustCodeTrans/evaluation/CodeBLEU/parser/my-languages.so', 'java'
        )
        for_while_transform = ForWhileTransformer(
            '/home/yangguang/PycharmProjects/RobustCodeTrans/evaluation/CodeBLEU/parser/my-languages.so', 'java'
        )
        df = pd.read_csv('dataset/test_j2p.csv')
        tgts = df['tgt'].tolist()
        srcs = df['src'].tolist()
        new_datas = []
        for i in tqdm(range(len(srcs))):
            # 将src代码进行语法转换并进行语义攻击
            code, meta = transform_java_code(model, srcs[i], i, operand_swap, operand_transform, if_else_transform, for_while_transform)
            code = re.sub("[ \t\n]+", " ", code)
            # 如果攻击成功，就将攻击后的src代码和原始tgt代码一起写入文件
            if meta=='success':
                count += 1
                new_datas.append([code, tgts[i]])
            # 如果攻击失败，就将原始src代码和原始tgt代码一起写入文件
            else:
                new_datas.append([srcs[i], tgts[i]])

        df = pd.DataFrame(new_datas, columns=['src', 'tgt'])
        df.to_csv("dataset_attack/syntax_"+task+"_"+model_type+".csv", index=False)
        return count
    if task == 'p2j':
        # a>b --> b<a
        operand_swap = OperandSwap(
            '/home/yangguang/PycharmProjects/RobustCodeTrans/evaluation/CodeBLEU/parser/my-languages.so', 'python'
        )
        # a+=b --> a=a+b
        operand_transform = ConfusionRemover(
            '/home/yangguang/PycharmProjects/RobustCodeTrans/evaluation/CodeBLEU/parser/my-languages.so', 'python'
        )
        if_else_transform = BlockSwap(
            '/home/yangguang/PycharmProjects/RobustCodeTrans/evaluation/CodeBLEU/parser/my-languages.so', 'python'
        )
        for_while_transform = ForWhileTransformer(
            '/home/yangguang/PycharmProjects/RobustCodeTrans/evaluation/CodeBLEU/parser/my-languages.so', 'python'
        )
        df = pd.read_csv('dataset/test_p2j.csv')
        tgts = df['tgt'].tolist()
        srcs = df['src'].tolist()
        new_datas = []
        for i in tqdm(range(len(srcs))):
            # 将src代码进行语法转换并进行语义攻击
            code, meta = transform_python_code(model, srcs[i], i, operand_swap, operand_transform, if_else_transform, for_while_transform)
            code = re.sub("[ \t\n]+", " ", code)
            # 如果攻击成功，就将攻击后的src代码和原始tgt代码一起写入文件
            if meta=='success':
                count += 1
                new_datas.append([code, tgts[i]])
            # 如果攻击失败，就将原始src代码和原始tgt代码一起写入文件
            else:
                new_datas.append([srcs[i], tgts[i]])

        df = pd.DataFrame(new_datas, columns=['src', 'tgt'])
        df.to_csv("dataset_attack/syntax_"+task+"_"+model_type+".csv", index=False)
        return count

if __name__ == '__main__':
    model_dict = {
        'codet5': '/home/yangguang/models/codet5-base',
        'plbart': '/home/yangguang/models/plbart-base',
        'natgen': '/home/yangguang/models/NatGen',
        'unixcoder': '/home/yangguang/models/unixcoder-base',
        'codegpt-java': '/home/yangguang/models/CodeGPT-small-java',
        'codegpt-py': '/home/yangguang/models/CodeGPT-small-py',
        'codegpt-adapter-java': '/home/yangguang/models/CodeGPT-small-java-adaptedGPT2',
        'codegpt-adapter-py': '/home/yangguang/models/CodeGPT-small-py-adaptedGPT2',
        'codebert': '/home/yangguang/models/codebert-base',
        'graphcodebert': '/home/yangguang/models/graphcodebert-base',
        'contrabert': '/home/yangguang/models/ContraBERT_G',
        'codegen': '/home/yangguang/models/codegen-350M-multi'
    }
    set_seed(1234)

    # task = 'j2p'
    # # for model_type in ['codet5', 'plbart', 'natgen', 'unixcoder', 'codegpt-py', 'codegpt-adapter-py', 'codebert', 'graphcodebert']:
    # for model_type in ['codebert', 'graphcodebert', 'contrabert']:
    #     generate_dataset_attack(model_dict, model_type, task)

    result_dict = {}
    task = 'p2j'
    for model_type in ['natgen', 'codet5', 'plbart', 'unixcoder', 'codebert', 'graphcodebert', 'contrabert', 'codegpt-java', 'codegpt-adapter-java', 'codegen']:
        count = generate_dataset_attack(model_dict, model_type, task)
        result_dict[model_type] = count
    # for model_type in ['codet5', 'plbart', 'natgen', 'unixcoder', 'codegpt-java', 'codegpt-adapter-java', 'codegen', 'codebert', 'graphcodebert', 'contrabert']:
    #     generate_dataset_attack(model_dict, model_type, task)
    df = pd.DataFrame([result_dict])
    df.to_csv("p2j_syntax.csv", index=False)