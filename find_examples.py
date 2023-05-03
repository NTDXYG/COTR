import os
import subprocess
import time

import pandas as pd
from pyflakes.api import checkPath
from tqdm import tqdm

from codegen.preprocessing.lang_processors.java_processor import JavaProcessor
from codegen.preprocessing.lang_processors.python_processor import PythonProcessor
from subprocess import run, check_output, CalledProcessError, STDOUT, PIPE

root_folder = "./third_party"
jprocessor = JavaProcessor(root_folder=root_folder)
pyprocessor = PythonProcessor(root_folder=root_folder)


def check_python_code_testcases(code, template_path):
    function_name = pyprocessor.get_function_name(code)
    data_list = []
    with open(template_path,'r',encoding='utf8') as f:
        for line in f.readlines():
            line = line.replace('#program#', code)
            line = line.replace('#function_name#', function_name)
            data_list.append(line)
    template_path = template_path.replace('txt', 'py')
    with open(template_path,'w',encoding='utf8') as fw:
        for data in data_list:
            fw.write(data)

    command = ["python", template_path]
    try:
        p = run(command, stderr=PIPE, timeout=5)
    except:
        return "Timeout"

    error_msg = p.stderr.decode("utf-8")
    os.remove(template_path)

    if len(error_msg) == 0:
        return True
    else:
        return False

def check_python_code(code):
    script = f"import numpy as np \nimport math\nfrom math import *\nimport collections\n" \
             f"from collections import *\nimport heapq\nimport itertools\nimport random\n" \
             f"import sys\n\n"
    with open("temp.py",'w',encoding='utf8') as f:
        f.write(script)
        f.write(code)
    result = checkPath('temp.py')
    os.remove('temp.py')
    if result == 1 or "may be undefined" in result:
        return False
    return True

def check_python_files(raw_input_file, attacked_input_file, raw_file, attacked_file):
    df = pd.read_csv(raw_file)
    raw_javas = df['src'].tolist()

    df = pd.read_csv(attacked_file)
    attacked_javas = df['src'].tolist()

    df = pd.read_csv(raw_input_file, header=None)
    raw_programs = df[0].tolist()

    df = pd.read_csv(attacked_input_file, header=None)
    attacked_programs = df[0].tolist()

    for i in tqdm(range(len(raw_programs))):
        raw_java = raw_javas[i].replace("Translate Java to Python: ", "")
        raw_java = jprocessor.detokenize_code(raw_java)

        attacked_java = attacked_javas[i].replace("Translate Java to Python: ", "")
        attacked_java = jprocessor.detokenize_code(attacked_java)

        raw_program = raw_programs[i]
        raw_program = pyprocessor.detokenize_code(raw_program)

        attacked_program = attacked_programs[i]
        attacked_program = pyprocessor.detokenize_code(attacked_program)
        # if (check_python_code(raw_program) == True and check_python_code(
        #         attacked_program) == False):
        #     print('攻击前的Java代码：')
        #     print(raw_java)
        #     print('攻击前生成的Python代码：')
        #     print(raw_program)
        #     print('攻击后的Java代码：')
        #     print(attacked_java)
        #     print('攻击后生成的Python代码：')
        #     print(attacked_program)
        #     print('-----------------------')

        if(check_python_code(raw_program) == True and check_python_code(attacked_program) == True and raw_program != attacked_program):
            raw_state = check_python_code_testcases(raw_program, 'test_cases_templates/'+str(i)+'.txt')
            attacked_state = check_python_code_testcases(attacked_program, 'test_cases_templates/'+str(i)+'.txt')
            if( raw_state == True and attacked_state != True):
                print('攻击前的Java代码：')
                print(raw_java)
                print('攻击前生成的Python代码：')
                print(raw_program)
                print('攻击后的Java代码：')
                print(attacked_java)
                print('攻击后生成的Python代码：')
                print(attacked_program)
                print('-----------------------')

if __name__ == '__main__':
    check_python_files('result/original/java-to-python/codet5.csv', 'result/attack/java-to-python/codet5.csv',
                       'dataset/test_j2p.csv', 'dataset_attack/syntax_j2p_codet5.csv')

