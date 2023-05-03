from evaluation.bleu import compute_bleu
from codegen.preprocessing.lang_processors.java_processor import JavaProcessor
from codegen.preprocessing.lang_processors.python_processor import PythonProcessor

root_folder = "../third_party"
jprocessor = JavaProcessor(root_folder=root_folder)
pyprocessor = PythonProcessor(root_folder=root_folder)

def get_bleu_socre(ref_file, hyp_file, task='j2p'):
    references = []
    with open(ref_file, 'r', encoding='utf-8') as fh:
        for line in fh:
            refs = [line.strip()]
            if task == 'p2j':
                refs = [jprocessor.detokenize_code(r) for r in refs]
            else:
                refs = [pyprocessor.detokenize_code(r) for r in refs]
            references.append([r.split() for r in refs])

    translations = []
    with open(hyp_file, 'r', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if task == 'p2j':
                line = jprocessor.detokenize_code(line)
            else:
                line = pyprocessor.detokenize_code(line)
            translations.append(line.split())

    assert len(references) == len(translations)
    count = 0
    for i in range(len(references)):
        refs = references[i]  # r is a list of 'list of tokens'
        # print(refs)
        t = translations[i]  # 'list of tokens'
        # print(t)
        for r in refs:
            if r == t:
                count += 1
                break
    acc = round(count / len(translations) * 100, 2)
    bleu_score, _, _, _, _, _ = compute_bleu(references, translations, 4, True)
    bleu_score = round(100 * bleu_score, 2)
    # print('BLEU:\t\t%.2f\nExact Match:\t\t%.2f' % (bleu_score, acc))
    return bleu_score, acc

# print(get_bleu_socre('../ref.txt', '../pred.txt', 'j2p'))