import datetime
import logging
from PLM import Decoder_Model
from utils import set_seed

set_seed(1234)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

model_dict = {
    'codegpt-java': '/home/yangguang/models/CodeGPT-small-java',
    'codegpt-py': '/home/yangguang/models/CodeGPT-small-py',
    'codegpt-adapter-java': '/home/yangguang/models/CodeGPT-small-java-adaptedGPT2',
    'codegpt-adapter-py': '/home/yangguang/models/CodeGPT-small-py-adaptedGPT2',
    'codegen': '/home/yangguang/models/codegen-350M-multi'
}

# model_type = 'codegen'
# task = 'j2p'
# result = model.predict("Translate Python to Java: def validPosition ( arr , N , K ) : NEW_LINE INDENT count = 0 ; sum = 0 ; NEW_LINE for i in range ( N ) : NEW_LINE INDENT sum += arr [ i ] ; NEW_LINE DEDENT for i in range ( N ) : NEW_LINE INDENT if ( ( arr [ i ] + K ) > ( sum - arr [ i ] ) ) : NEW_LINE INDENT count += 1 ; NEW_LINE DEDENT DEDENT return count ; NEW_LINE DEDENT")

# print(result)

for model_type in ['codegpt-java', 'codegpt-adapter-java', 'codegen']:
    task = 'p2j'
    # 初始化模型
    model = Decoder_Model(model_type=model_type, model_name_or_path=model_dict[model_type], load_model_path=None,
                          block_size=700,
                          beam_size=10, max_source_length=350, max_target_length=700)

    start = datetime.datetime.now()

    # 模型训练
    model.train(train_filename='dataset_defense/train_' + task + '.csv', train_batch_size=4, learning_rate=5e-5,
                num_train_epochs=50, early_stop=5, task=task, do_eval=True,
                eval_filename='dataset/valid_' + task + '.csv',
                eval_batch_size=1, output_dir='models/DA_AT/valid_output_' + task + '/' + model_type + '/',
                do_eval_bleu=True, AT=True)

    end = datetime.datetime.now()
    print(end - start)

    # 加载微调过后的模型参数
    model = Decoder_Model(model_type=model_type, model_name_or_path=model_dict[model_type], beam_size=10,
                          block_size=700,
                          max_source_length=350, max_target_length=700,
                          load_model_path='models/DA_AT/valid_output_' + task + '/' + model_type + '/checkpoint-best-bleu/pytorch_model.bin')

    model.test(batch_size=1, filename='dataset/test_' + task + '.csv',
               output_dir='models/DA_AT/test_output_' + task + '/' + model_type + '/', task=task)

    model.test(batch_size=1, filename='dataset_attack/syntax_' + task + '_' + model_type + '.csv',
               output_dir='result/DA_AT_RP1/python-to-java/', task=task)

for model_type in ['codegpt-py', 'codegpt-adapter-py', 'codegen']:
    task = 'j2p'
    # 初始化模型
    model = Decoder_Model(model_type=model_type, model_name_or_path=model_dict[model_type], load_model_path=None,
                          block_size=700,
                          beam_size=10, max_source_length=350, max_target_length=700)

    start = datetime.datetime.now()

    # 模型训练
    model.train(train_filename='dataset_defense/train_' + task + '.csv', train_batch_size=4, learning_rate=5e-5,
                num_train_epochs=50, early_stop=5, task=task, do_eval=True,
                eval_filename='dataset/valid_' + task + '.csv',
                eval_batch_size=1, output_dir='models/DA_AT/valid_output_' + task + '/' + model_type + '/',
                do_eval_bleu=True, AT=True)

    end = datetime.datetime.now()
    print(end - start)

    # 加载微调过后的模型参数
    model = Decoder_Model(model_type=model_type, model_name_or_path=model_dict[model_type], beam_size=10,
                          block_size=700,
                          max_source_length=350, max_target_length=700,
                          load_model_path='models/DA_AT/valid_output_' + task + '/' + model_type + '/checkpoint-best-bleu/pytorch_model.bin')

    model.test(batch_size=1, filename='dataset/test_' + task + '.csv',
               output_dir='models/DA_AT/test_output_' + task + '/' + model_type + '/', task=task)

    model.test(batch_size=1, filename='dataset_attack/syntax_' + task + '_' + model_type + '.csv',
               output_dir='result/DA_AT_RP1/java-to-python/', task=task)