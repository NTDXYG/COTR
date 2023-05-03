import datetime
import logging
from PLM import Encoder_Decoder
from utils import set_seed

set_seed(1234)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

model_dict = {
    'codet5': '/home/yangguang/models/codet5-base',
    'plbart': '/home/yangguang/models/plbart-base',
    'natgen': '/home/yangguang/models/NatGen',
}

# for model_type in ['natgen', 'plbart']:
#     task = 'j2p'
#     # 初始化模型
#     model = Encoder_Decoder(model_type=model_type, model_name_or_path=model_dict[model_type], load_model_path=None,
#                       beam_size=10, max_source_length=350, max_target_length=350)
#
#     start = datetime.datetime.now()
#
#     # 模型训练
#     model.train(train_filename = 'dataset/train_'+task+'.csv', train_batch_size = 6, learning_rate = 5e-5,
#                 num_train_epochs = 50, early_stop = 5, task=task, do_eval=True, eval_filename='dataset/valid_'+task+'.csv',
#                 eval_batch_size=12, output_dir='models/AT/valid_output_'+task+'/'+model_type+'/', do_eval_bleu=True, AT=True)
#
#     end = datetime.datetime.now()
#     print(end-start)
#
#     # 加载微调过后的模型参数
#     model = Encoder_Decoder(model_type=model_type, model_name_or_path=model_dict[model_type], beam_size=10,
#                       max_source_length=350, max_target_length=350,
#                       load_model_path='models/AT/valid_output_'+task+'/'+model_type+'/checkpoint-best-bleu/pytorch_model.bin')
#
#     model.test(batch_size=1, filename='dataset/test_'+task+'.csv', output_dir='models/AT/test_output_'+task+'/'+model_type+'/', task = task)
#
#     model.test(batch_size=1, filename='dataset_attack/syntax_'+task+'_'+model_type+'.csv',
#                output_dir='result/AT_RP1/java-to-python/', task = task)
#
# for model_type in ['natgen', 'codet5', 'plbart']:
#     task = 'p2j'
#     # 初始化模型
#     model = Encoder_Decoder(model_type=model_type, model_name_or_path=model_dict[model_type], load_model_path=None,
#                             beam_size=10, max_source_length=350, max_target_length=350)
#
#     start = datetime.datetime.now()
#
#     # 模型训练
#     model.train(train_filename='dataset/train_' + task + '.csv', train_batch_size=6, learning_rate=5e-5,
#                 num_train_epochs=50, early_stop=5, task=task, do_eval=True,
#                 eval_filename='dataset/valid_' + task + '.csv',
#                 eval_batch_size=12, output_dir='models/AT/valid_output_' + task + '/' + model_type + '/',
#                 do_eval_bleu=True, AT=True)
#
#     end = datetime.datetime.now()
#     print(end - start)
#
#     # 加载微调过后的模型参数
#     model = Encoder_Decoder(model_type=model_type, model_name_or_path=model_dict[model_type], beam_size=10,
#                             max_source_length=350, max_target_length=350,
#                             load_model_path='models/AT/valid_output_' + task + '/' + model_type + '/checkpoint-best-bleu/pytorch_model.bin')
#
#     model.test(batch_size=1, filename='dataset/test_' + task + '.csv',
#                output_dir='models/AT/test_output_' + task + '/' + model_type + '/', task=task)
#
#     model.test(batch_size=1, filename='dataset_attack/syntax_' + task + '_' + model_type + '.csv',
#                output_dir='result/AT_RP1/python-to-java/', task=task)

for model_type in ['codet5', 'plbart']:
    for task in ['j2p', 'p2j']:
        # 初始化模型
        model = Encoder_Decoder(model_type=model_type, model_name_or_path=model_dict[model_type], load_model_path=None,
                                beam_size=5, max_source_length=350, max_target_length=350)

        start = datetime.datetime.now()

        # 模型训练
        model.train(train_filename='dataset_defense/train_' + task + '.csv', train_batch_size=6, learning_rate=5e-5,
                    num_train_epochs=50, early_stop=5, task=task, do_eval=True,
                    eval_filename='dataset/valid_' + task + '.csv',
                    eval_batch_size=12, output_dir='models/DA/valid_output_' + task + '/' + model_type + '/',
                    do_eval_bleu=True, AT=False)

        end = datetime.datetime.now()
        print(end - start)

        # 加载微调过后的模型参数
        model = Encoder_Decoder(model_type=model_type, model_name_or_path=model_dict[model_type], beam_size=5,
                                max_source_length=350, max_target_length=350,
                                load_model_path='models/DA/valid_output_' + task + '/' + model_type + '/checkpoint-best-bleu/pytorch_model.bin')

        model.test(batch_size=1, filename='dataset/test_' + task + '.csv',
                   output_dir='models/DA/test_output_' + task + '/' + model_type + '/', task=task)
        if task == 'j2p':
            model.test(batch_size=1, filename='dataset_attack/syntax_' + task + '_' + model_type + '.csv',
                   output_dir='result/DA_RP1/java-to-python/', task=task)
        else:
            model.test(batch_size=1, filename='dataset_attack/syntax_' + task + '_' + model_type + '.csv',
                       output_dir='result/DA_RP1/python-to-java/', task=task)

for model_type in ['natgen', 'codet5', 'plbart']:
    for task in ['j2p', 'p2j']:
        # 初始化模型
        model = Encoder_Decoder(model_type=model_type, model_name_or_path=model_dict[model_type], load_model_path=None,
                                beam_size=10, max_source_length=350, max_target_length=350)

        start = datetime.datetime.now()

        # 模型训练
        model.train(train_filename='dataset_defense/train_' + task + '.csv', train_batch_size=6, learning_rate=5e-5,
                    num_train_epochs=50, early_stop=5, task=task, do_eval=True,
                    eval_filename='dataset/valid_' + task + '.csv',
                    eval_batch_size=12, output_dir='models/DA_AT/valid_output_' + task + '/' + model_type + '/',
                    do_eval_bleu=True, AT=True)

        end = datetime.datetime.now()
        print(end - start)

        # 加载微调过后的模型参数
        model = Encoder_Decoder(model_type=model_type, model_name_or_path=model_dict[model_type], beam_size=10,
                                max_source_length=350, max_target_length=350,
                                load_model_path='models/DA_AT/valid_output_' + task + '/' + model_type + '/checkpoint-best-bleu/pytorch_model.bin')

        model.test(batch_size=1, filename='dataset/test_' + task + '.csv',
                   output_dir='models/DA_AT/test_output_' + task + '/' + model_type + '/', task=task)
        if task == 'j2p':
            model.test(batch_size=1, filename='dataset_attack/syntax_' + task + '_' + model_type + '.csv',
                       output_dir='result/DA_AT_RP1/java-to-python/', task=task)
        else:
            model.test(batch_size=1, filename='dataset_attack/syntax_' + task + '_' + model_type + '.csv',
                       output_dir='result/DA_AT_RP1/python-to-java/', task=task)
# model.predict("Translate Python to Java: def validPosition ( arr , N , K ) : NEW_LINE INDENT count = 0 ; sum = 0 ; NEW_LINE for i in range ( N ) : NEW_LINE INDENT sum += arr [ i ] ; NEW_LINE DEDENT for i in range ( N ) : NEW_LINE INDENT if ( ( arr [ i ] + K ) > ( sum - arr [ i ] ) ) : NEW_LINE INDENT count += 1 ; NEW_LINE DEDENT DEDENT return count ; NEW_LINE DEDENT")