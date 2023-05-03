## The functions of the folders:

1. codegen: stores preprocessing and reconstruction data for code.
2. dataset: stores datasets, including training, validation, and testing datasets.
3. dataset_attack: stores adversarial sample datasets for each model under each task.
4. dataset_defense: stores datasets after data augmentation (DA).
5. evaluation: stores evaluation metrics, including BLEU and CodeBLEU.
6. human: stores manually evaluated code.
7. models: stores model weights (uploaded after the article is accepted).
8. program_transformer: stores code for model transformation, including syntax tree transformation and identifier renaming.
9. pyflakes: used to check for syntax errors in generated Python code (not utilized).
10. result: stores results (uploaded after the article is accepted).
11. test_cases_templates: stores templates for test cases.
12. third_party: stores the third-party tree-sitter library.

## The functions of the files:

1. attack_radar_w2v.py: trains the word2vec model required for the radar attack. attack_radar.py implements the radar attack algorithm.
2. attack.py: implements the attack algorithm proposed in our paper.
3. build.py and calc_code_bleu.py: extract and calculate CodeBLEU.
4. compile.py: calculates Code Exec and Pass@1 metrics.
5. datasets.py: processes model input data.
6. defense.py: implements the sampling-based data augmentation algorithm.
7. EncModel.py and UniModel.py: respectively implement the Encoder-only pre-training model and UniXcoder model.
8. find_examples.py: finds motivation example figures, such as Figure 1 in the paper.
9. NPGD.py: implements adversarial training with the PGD algorithm.
10. PLM.py: custom code for training, validating, and inferring aggregate models.
11. utils.py: toolset.
12. run_enc.py, run_dec.py, run_unixcoder.py, and run_enc_dec.py: respectively run Encoder-only, Decoder-only, UniXcoder, and Encoder-Decoder models. Various fine-tuning can be performed by modifying the parameters in these files.