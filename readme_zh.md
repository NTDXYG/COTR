文件夹的作用：
----------------
1. codegen 用来存放针对代码数据的预处理与还原
2. dataset 用来存放数据集，包括训练集、验证集、测试集
3. dataset_attack 用来存放每个任务下每个模型的对抗样本数据集
4. dataset_defense 用来存放DA后的数据集
5. evaluation 用来存放评价指标，包括BLEU、CodeBLEU
6. human 用来存放人工评估的代码
7. models 用来存放模型权重 （文章录用后上传）
8. program_transformer 用来模型转换的代码，包括语法树的转换和标识符的重命名
9. pyflakes 用来检查生成的python代码的语法错误（没利用）
10. result 用来存放结果 （文章录用后上传）
11. test_cases_templates 用来存放测试用例的模板
12. third_party 用来存放第三方tree-sitter库

文件的作用：
----------------

1. attack_radar_w2v.py 用来训练radar攻击所需要的word2vec模型；attack_radar.py是radar攻击的算法实现。
2. attack.py是我们论文中提出的攻击算法实现。
3. build.py & calc_code_bleu.py 提取出来计算codebleu的。
4. compile.py 用来计算Code Exec和Pass@1指标的。
5. datasets.py 用来处理模型输入的数据的。
6. defense.py 用来实现基于采样的数据增强算法的。
7. EncModel.py 和 UniModel.py 分别实现Encoder-only的预训练模型和UniXcoder模型的。
8. find_examples.py 用来找寻动机例图的，如文中的图1。
9. NPGD.py 实现对抗训练PGD算法的。
10. PLM.py 自定义的聚合模型训练、验证和推理的代码。
11. utils.py 工具包
12. run_enc.py, run_dec.py, run_unixcoder.py和run_enc_dec.py分别是运行Encoder-only, Decoder-only, UniXcoder和Encoder-Decoder模型的。通过修改里面的参数进行各种微调。 