import pandas as pd

from codegen.preprocessing.lang_processors.java_processor import JavaProcessor
from codegen.preprocessing.lang_processors.python_processor import PythonProcessor

pyprocessor = PythonProcessor()
jprocessor = JavaProcessor("/home/yangguang/PycharmProjects/RobustCodeTrans/third_party")


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

def find_human(task, model_type):
    count = 0

    df = pd.read_csv("../dataset/test_"+task+".csv")
    origin_src = df['src'].tolist()

    df = pd.read_csv("../dataset_attack/radar_"+task+"_"+model_type+".csv")
    radar_src = df['src'].tolist()

    df = pd.read_csv("../dataset_attack/syntax_"+task+"_"+model_type+".csv")
    syntax_src = df['src'].tolist()

    for i in range(len(origin_src)):
        if origin_src[i] != radar_src[i] and origin_src[i] != syntax_src[i]:
            origin = origin_src[i].replace("Translate Python to Java: ", "")
            origin = decode(origin, 'python')
            # print('origin:')
            # print(origin)

            radar = radar_src[i].replace("Translate Python to Java: ", "")
            radar = decode(radar, 'python')
            # print('radar:')
            # print(radar)

            syntax = syntax_src[i].replace("Translate Python to Java: ", "")
            syntax = decode(syntax, 'python')
            # print('syntax:')
            # print(syntax)
            count += 1
    return count

c = 0
for model in ['natgen', 'codet5', 'plbart', 'unixcoder', 'codebert', 'graphcodebert', 'contrabert', 'codegpt-py', 'codegpt-adapter-py', 'codegen']:
    c += find_human('j2p', model)
for model in ['natgen', 'codet5', 'plbart', 'unixcoder', 'codebert', 'graphcodebert', 'contrabert', 'codegpt-java', 'codegpt-adapter-java', 'codegen']:
    c += find_human('p2j', model)
print(c)

# def ofOccurrences(str, substr):
#     counter = 0
#     for i in range(0, len(str)):
#         if (str[i] == substr[0]):
#             for j in range(i + 1, len(str)):
#                 if (str[j] == substr[1]):
#                     for k in range(j + 1, len(str)):
#                         if (str[k] == substr[2]):
#                             counter = counter + 1
#     return counter
#
#
# def findOccurrences(str, substr):
#     counter = 0
#     for i in range(0, len(str)):
#         if (str[i] == substr[0]):
#             for j in range(i + 1, len(str)):
#                 if (str[j] == substr[1]):
#                     for k in range(j + 1, len(str)):
#                         if (substr[2] == str[k]):
#                             counter = counter + 1
#     return counter
