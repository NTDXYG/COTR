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
    # print(code)
    # print(function_name)
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
        # print(error_msg)
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

def check_python_files(input_file):
    df = pd.read_csv(input_file, header=None)
    programs = df[0].tolist()
    # df = pd.read_csv(input_file)
    # programs = df['tgt'].tolist()

    syntax_success, syntax_error, success_testcases, error_testcases = 0, 0, 0, 0
    timeout_error = 0

    # for i in tqdm(range(25, 50)):
    for i in tqdm(range(len(programs))):
        program = programs[i]
        program = pyprocessor.detokenize_code(program)
        if(check_python_code(program) == False):
            syntax_error += 1
        else:
            syntax_success += 1
            state = check_python_code_testcases(program, 'test_cases_templates/'+str(i)+'.txt')
            if( state == True):
                success_testcases += 1
            elif("Timeout" == state):
                timeout_error += 1
            else:
                error_testcases += 1

    return ('SyntaxErrors - {}, SyntaxSuccess - {} [SuccessRun-TestCases - {}, ErrorsRun-TestCases - {}, Timeout-TestCases - {}]'.format(
        syntax_error, syntax_success, success_testcases, error_testcases, timeout_error)
    )


def check_java_code_testcases(code, template_path):
    function_name = jprocessor.get_function_name(code)

    data_list = []
    with open("./test_cases_templates/"+template_path, 'r', encoding='utf8') as f:
        for line in f.readlines():
            line = line.replace('#program#', code)
            line = line.replace('#function_name#', function_name)
            data_list.append(line)
    template_path = template_path.replace('txt', 'java')
    with open("./test_cases_templates/"+template_path, 'w', encoding='utf8') as fw:
        for data in data_list:
            fw.write(data)

    command = ["javac", "./test_cases_templates/"+template_path]
    try:
        check_output(command)
    except:
        os.remove("./test_cases_templates/"+template_path)
        return False
    os.remove("./test_cases_templates/"+template_path)
    command = ["java", "-cp", "./test_cases_templates", template_path.replace(".java", "")]
    # print(command)
    try:
        # result, stderr = proc.communicate(timeout=5)
        p = run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
        result = str(p.stdout)
        if 'Fail' in result or 'False' in result:
            return False
        if 'Fail' not in result and 'False' not in result:
            return True
    except:
        return "Timeout"

    os.remove("./test_cases_templates/"+template_path.replace(".java", ".class"))
    return False



def check_java_code(code):
    program = 'import java.util.*;\n' \
              'import java.util.stream.*;\n' \
              'import java.lang.*;\n'
    program += '\n'
    program += 'public class temp { \n\n'
    program += code
    program += '}\n'
    with open("temp.java", 'w', encoding='utf8') as f:
        f.write(program)
    command = ["javac", "temp.java"]
    try:
        check_output(command)
        os.remove("temp.java")
        os.remove("temp.class")
        return True
    except:
        os.remove("temp.java")
        return False


def check_java_files(input_file):
    df = pd.read_csv(input_file, header=None)
    programs = df[0].tolist()
    # df = pd.read_csv(input_file)
    # programs = df['tgt'].tolist()

    syntax_success, syntax_error, success_testcases, error_testcases = 0, 0, 0, 0
    timeout_error = 0

    for i in tqdm(range(len(programs))):
        program = str(programs[i])
        program = jprocessor.detokenize_code(program)
        if(check_java_code(program) == False):
            syntax_error += 1
        else:
            syntax_success += 1
            try:
                state = check_java_code_testcases(program, 'Main_'+str(i)+'.txt')
                if(state == True):
                    success_testcases += 1
                elif("Timeout" == state):
                    timeout_error += 1
                else:
                    error_testcases += 1
            except:
                error_testcases += 1

    return ('SyntaxErrors - {}, SyntaxSuccess - {} [SuccessRun-TestCases - {}, ErrorsRun-TestCases - {}, Timeout-TestCases - {}]'.format(
        syntax_error, syntax_success, success_testcases, error_testcases, timeout_error)
    )


if __name__ == '__main__':
    # result = check_python_files('result/syntax-attack-java-to-python/contrabert.csv')
    # print(result)

    # result = check_java_files('result/python-to-java/contrabert.csv')
    # print(result)

    # result_dict = {}
    # for model in ['natgen', 'codet5', 'plbart', 'unixcoder', 'codebert', 'graphcodebert', 'codegpt', 'codegpt_adapter']:
    #     result = check_python_files('result/java-to-python/'+model+'.csv')
    #     result_dict[model] = result
    # print(result_dict)

    # result_dict = {}
    # for model in ['natgen', 'codet5', 'plbart', 'unixcoder', 'codebert', 'graphcodebert', 'codegpt', 'codegpt_adapter']:
    #     result = check_python_files('result/syntax-attack-java-to-python/'+model+'.csv')
    #     result_dict[model] = result
    # print(result_dict)

    # result_dict = {}
    # for model in ['natgen', 'codet5', 'plbart', 'unixcoder', 'codebert', 'graphcodebert', 'codegpt', 'codegpt_adapter']:
    #     result = check_java_files('result/python-to-java/'+model+'.csv')
    #     result_dict[model] = result
    # print(result_dict)
    # defense_original(Pass@1)
    # defense_sampling_RP@1
    # result_dict = {}
    # for model in ['contrabert']:
    #     result = check_python_files('result/DA_AT_Pass1/java-to-python/'+model+'.csv')
    #     result_dict[model] = result
    # print(result_dict)

    result_dict = {}
    for model in ['contrabert']:
        result = check_java_files('result/DA_AT_RP1/python-to-java/'+model+'.csv')
        result_dict[model] = result
    print(result_dict)

    # result = check_java_files('result/original/python-to-java/natgen.csv')
    # print(result)
    # result_dict = {}

    # result = check_python_files('result/attack/java-to-python/codebert.csv')
    # result_dict['codebert'] = result
    #
    # result = check_python_files('result/attack/java-to-python/graphcodebert.csv')
    # result_dict['graphcodebert'] = result

    # result = check_python_files('result/defense_AT_original/java-to-python/plbart.csv')
    # result_dict['defense_original'] = result
    #
    # result = check_python_files('result/defense_AT/java-to-python/plbart.csv')
    # result_dict['defense'] = result
    #
    # print(result_dict)