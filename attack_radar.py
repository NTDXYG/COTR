import re
from string import digits, ascii_lowercase

from sko.GA import GA
import pandas as pd
import random

from gensim.models import Word2Vec
from sko.tools import set_run_mode
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

def gen_random_delete_char(word):
    temp = random.randint(1, len(word))
    return word[:temp] + word[temp+1:]

def gen_random_swap_char(word):
    if(len(word)>2 and word != len(word) * word[0]):
        try:
            temp = random.randint(1, len(word)-1)
            return word[:temp] + word[temp+1] + word[temp] + word[temp+2:]
        except:
            return word
    else:
        return word

def gen_simi_replace_char(word):
    result = word
    for w in word:
        if(w == '2'):
            result.replace(w, 'to')
        if (w == '4'):
            result.replace(w, 'for')
        if (w == 'l'):
            result.replace(w, '1')
        if (w == 'o'):
            result.replace(w, '0')
        if (w == 'q'):
            result.replace(w, '9')
        if (w == 's'):
            result.replace(w, '5')
    return result

def snake_case(s):
    return re.sub('([A-Z])', r'_\1', s).lstrip('_')

def to_lower_camle_case(x):
    s = re.sub('_([a-zA-Z])', lambda m: (m.group(1).upper()), x)
    return s[0].lower() + s[1:]

def get_python_score(model, x, i):
    p = model.predict("Translate Java to Python: " + encode(x, 'java'))
    # print(p)
    program = pyprocessor.detokenize_code(p)
    # print(program)
    try:
        state = check_python_code_testcases(program, 'test_cases_templates/' + str(i) + '.txt')
    except:
        state = False
    # 如果破坏语义，则返回对抗样本
    if (state == True):
        return 1
    else:
        return 0

def get_java_score(model, x, i):
    p = model.predict("Translate Python to Java: " + encode(x, 'python'))
    # print(p)
    program = pyprocessor.detokenize_code(p)
    # print(program)
    try:
        state = check_java_code_testcases(program, 'Main_'+str(i)+'.txt')
    except:
        state = False
    # 如果破坏语义，则返回对抗样本
    if (state == True):
        return 1
    else:
        return 0

def transform_java_code(model, java, i, model_path):
    # print('开始转换...')
    w2v_model = Word2Vec.load(model_path)
    # 先检查为攻击前model生成的java代码是否能够通过测试用例，如果能通过，则直接跳过
    code = model.predict(java)
    program = pyprocessor.detokenize_code(code)
    try:
        state = check_python_code_testcases(program, 'test_cases_templates/' + str(i) + '.txt')
    except:
        state = False
    if state == False:
        code, meta = "", "fail"
        print('原始生成的代码就不能通过测试用例，直接返回')
        return code, meta
    # 如果不能通过，则进行语法转换
    code, meta = '', ''
    decode_code = decode(java.replace("Translate Java to Python: ", ""), 'java')

    func_name = decode_code.split()[decode_code.split().index('(') - 1]
    raw_x = func_name
    temp_list = []

    if '_' not in func_name:
        func_name = snake_case(func_name)

    lis = [[] for q in range(len(func_name.split('_')))]
    for j in range(len(func_name.split('_'))):
        name = func_name.split('_')[j]
        sims = w2v_model.wv.most_similar(name, topn=20)  # get other similar words
        sims_filter = [k[0] for k in sims if k[0].isalpha()]
        for k in sims_filter[:2]:
            lis[j].append(k)
        lis[j].append(gen_random_delete_char(name))
        lis[j].append(gen_random_swap_char(name))
        lis[j].append(gen_simi_replace_char(name))

    def schaffer(p):
        attack = []
        for t in range(len(p)):
            attack.append(lis[t][int(p[t])])
        x_attack = to_lower_camle_case('_'.join(attack))
        attack_code = decode_code.replace(raw_x, x_attack)
        score = get_python_score(model, attack_code, i)
        return score

    set_run_mode(schaffer, 'cached')
    ga = GA(func=schaffer, n_dim=len(func_name.split('_')), size_pop=10, max_iter=50, prob_mut=0.001, lb=[0 for i in range(len(func_name.split('_')))], ub=[4 for i in range(len(func_name.split('_')))], precision=1, early_stop=3)
    best_x, best_y = ga.run()
    for i in range(len(best_x)):
        temp_list.append(lis[i][int(best_x[i])])

    if(best_y[0] == 0):
        print('攻击成功')
        attack_code = decode_code.replace(raw_x, to_lower_camle_case('_'.join(temp_list)))
        print(attack_code)
        code, meta = "Translate Java to Python: " + encode(attack_code, 'java'), "success"
        return code, meta
    else:
        return decode_code, "fail"

def transform_python_code(model, python, i, model_path):
    print('开始转换...')
    # 先检查为攻击前model生成的java代码是否能够通过测试用例，如果能通过，则直接跳过
    code = model.predict(python)
    program = jprocessor.detokenize_code(code)
    try:
        state = check_java_code_testcases(program, 'Main_'+str(i)+'.txt')
    except:
        state = False
    print(state)
    if state == False:
        code, meta = "", "fail"
        print('原始生成的代码就不能通过测试用例，直接返回')
        return code, meta
    # 如果不能通过，则进行语法转换
    code, meta = '', ''
    w2v_model = Word2Vec.load(model_path)
    decode_code = decode(python.replace("Translate Python to Java: ", ""), 'python')

    func_name = decode_code.split()[decode_code.split().index('def') + 1]
    # print(func_name)
    raw_x = func_name
    temp_list = []

    if '_' not in func_name:
        func_name = snake_case(func_name)

    lis = [[] for q in range(len(func_name.split('_')))]
    for j in range(len(func_name.split('_'))):
        name = func_name.split('_')[j]
        # print(name)
        sims = w2v_model.wv.most_similar(name, topn=10)  # get other similar words
        sims_filter = [k[0] for k in sims if k[0].isalpha()]
        for k in sims_filter[:2]:
            lis[j].append(k)
        lis[j].append(gen_random_delete_char(name))
        lis[j].append(gen_random_swap_char(name))
        lis[j].append(gen_simi_replace_char(name))

    def schaffer(p):
        attack = []
        for t in range(len(p)):
            attack.append(lis[t][int(p[t])])
        x_attack = to_lower_camle_case('_'.join(attack))
        attack_code = decode_code.replace(raw_x, x_attack)
        score = get_java_score(model, attack_code, i)
        return score

    set_run_mode(schaffer, 'cached')
    ga = GA(func=schaffer, n_dim=len(func_name.split('_')), size_pop=6, max_iter=20, prob_mut=0.001,
            lb=[0 for i in range(len(func_name.split('_')))], ub=[4 for i in range(len(func_name.split('_')))],
            precision=1, early_stop=3)
    best_x, best_y = ga.run()
    for i in range(len(best_x)):
        temp_list.append(lis[i][int(best_x[i])])

    if (best_y[0] == 0):
        print('攻击成功')
        attack_code = decode_code.replace(raw_x, to_lower_camle_case('_'.join(temp_list)))
        print(attack_code)
        code, meta = "Translate Python to Java: " + encode(attack_code, 'python'), "success"
        return code, meta
    else:
        return decode_code, "fail"

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
        model_path = 'models/word2vec.model'
        df = pd.read_csv('dataset/test_j2p.csv')
        tgts = df['tgt'].tolist()
        srcs = df['src'].tolist()
        new_datas = []
        for i in tqdm(range(len(srcs))):
            # 将src代码进行语法转换并进行语义攻击
            code, meta = transform_java_code(model, srcs[i], i, model_path)
            code = re.sub("[ \t\n]+", " ", code)
            # 如果攻击成功，就将攻击后的src代码和原始tgt代码一起写入文件
            if meta=='success':
                count += 1
                new_datas.append([code, tgts[i]])
            # 如果攻击失败，就将原始src代码和原始tgt代码一起写入文件
            else:
                new_datas.append([srcs[i], tgts[i]])
        df = pd.DataFrame(new_datas, columns=['src', 'tgt'])
        df.to_csv("dataset_attack/radar_"+task+"_"+model_type+".csv", index=False)
        return count

    if task == 'p2j':
        model_path = 'models/word2vec.model'
        df = pd.read_csv('dataset/test_p2j.csv')
        tgts = df['tgt'].tolist()
        srcs = df['src'].tolist()
        new_datas = []
        for i in tqdm(range(len(srcs))):
            # 将src代码进行语法转换并进行语义攻击
            code, meta = transform_python_code(model, srcs[i], i, model_path)
            code = re.sub("[ \t\n]+", " ", code)
            # 如果攻击成功，就将攻击后的src代码和原始tgt代码一起写入文件
            if meta=='success':
                count += 1
                new_datas.append([code, tgts[i]])
            # 如果攻击失败，就将原始src代码和原始tgt代码一起写入文件
            else:
                new_datas.append([srcs[i], tgts[i]])

        df = pd.DataFrame(new_datas, columns=['src', 'tgt'])
        df.to_csv("dataset_attack/radar_"+task+"_"+model_type+".csv", index=False)
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

    result_dict = {}
    task = 'j2p'
    # for model_type in ['codet5', 'plbart', 'natgen', 'unixcoder', 'codegpt-py', 'codegpt-adapter-py', 'codebert', 'graphcodebert']:
    for model_type in ['natgen', 'codet5', 'plbart', 'unixcoder', 'codebert', 'graphcodebert', 'contrabert', 'codegpt-py', 'codegpt-adapter-py', 'codegen']:
        count = generate_dataset_attack(model_dict, model_type, task)
        result_dict[model_type] = count
    # print(result_dict)
    df = pd.DataFrame([result_dict])
    df.to_csv("j2p_radar.csv", index=False)

    result_dict = {}
    task = 'p2j'
    # for model_type in ['codet5', 'plbart', 'natgen', 'unixcoder', 'codegpt-py', 'codegpt-adapter-py', 'codebert', 'graphcodebert']:
    for model_type in ['natgen', 'codet5', 'plbart', 'unixcoder', 'codebert', 'graphcodebert', 'contrabert', 'codegpt-java', 'codegpt-adapter-java', 'codegen']:
        count = generate_dataset_attack(model_dict, model_type, task)
        result_dict[model_type] = count
    # print(result_dict)
    df = pd.DataFrame([result_dict])
    df.to_csv("p2j_radar.csv", index=False)