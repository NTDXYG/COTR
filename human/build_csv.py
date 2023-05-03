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
    result = []

    df = pd.read_csv("../dataset/test_"+task+".csv")
    origin_src = df['src'].tolist()

    df = pd.read_csv("../dataset_attack/radar_"+task+"_"+model_type+".csv")
    radar_src = df['src'].tolist()

    df = pd.read_csv("../dataset_attack/syntax_"+task+"_"+model_type+".csv")
    syntax_src = df['src'].tolist()

    if task == 'p2j':
        for i in range(len(origin_src)):
            if origin_src[i] != radar_src[i] and origin_src[i] != syntax_src[i]:
                origin = origin_src[i].replace("Translate Python to Java: ", "")
                origin = decode(origin, 'python')
                radar = radar_src[i].replace("Translate Python to Java: ", "")
                radar = decode(radar, 'python')
                syntax = syntax_src[i].replace("Translate Python to Java: ", "")
                syntax = decode(syntax, 'python')
                syntaxs = syntax.split('\n')
                syntax = '\n'.join([s for s in syntaxs if len(s.strip())>0])
                result.append([origin, radar, syntax])
    else:
        for i in range(len(origin_src)):
            if origin_src[i] != radar_src[i] and origin_src[i] != syntax_src[i]:
                origin = origin_src[i].replace("Translate Java to Python: ", "")
                origin = decode(origin, 'java')
                radar = radar_src[i].replace("Translate Java to Python: ", "")
                radar = decode(radar, 'java')
                syntax = syntax_src[i].replace("Translate Java to Python: ", "")
                syntax = decode(syntax, 'java')
                syntaxs = syntax.split('\n')
                syntax = '\n'.join([s for s in syntaxs if len(s.strip())>0])
                # syntax = syntax.replace('\n\n', '')
                result.append([origin, radar, syntax])
    return result

results = []
for model in ['natgen', 'codet5', 'plbart', 'unixcoder', 'codebert', 'graphcodebert', 'contrabert', 'codegpt-py', 'codegpt-adapter-py', 'codegen']:
    results.extend(find_human('j2p', model))
for model in ['natgen', 'codet5', 'plbart', 'unixcoder', 'codebert', 'graphcodebert', 'contrabert', 'codegpt-java', 'codegpt-adapter-java', 'codegen']:
    results.extend(find_human('p2j', model))
print(len(results))
df = pd.DataFrame(results, columns=['original', 'radar', 'syntax'])
df.to_csv('Human.csv')
