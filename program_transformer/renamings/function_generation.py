import math
import random
import re
from typing import Union, Tuple
import os

from program_transformer.language_processors import (
    JavaAndCPPProcessor,
    PythonProcessor,
)
from program_transformer.language_processors.utils import get_tokens
from program_transformer.transformations import TransformationBase
import os
import torch
from unixcoder import UniXcoder

processor_function = {
    "java": JavaAndCPPProcessor,
    "python": PythonProcessor,
}

tokenizer_function = {
    "java": get_tokens,
    "python": PythonProcessor.get_tokens,
}


class FuncNameGen(TransformationBase):
    def __init__(
            self,
            parser_path: str,
            language: str,
            model_path
    ):
        super(FuncNameGen, self).__init__(
            parser_path=parser_path,
            language=language
        )
        self.language = language
        self.model_path = model_path
        self.processor = processor_function[self.language]
        self.tokenizer_function = tokenizer_function[self.language]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = UniXcoder(model_path)
        self.model.to(self.device)

        self.not_var_ptype = ["function_declarator", "class_declaration", "method_declaration", "function_definition",
                              "function_declaration", "call", "local_function_statement"]

    def extract_func_name(self, code_string):
        if self.language == 'java':
            if(code_string[code_string.index("(") - 1] == ''):
                return code_string[code_string.index("(") - 2]
            return code_string[code_string.index("(") - 1]
        if self.language == 'python':
            return code_string[code_string.index("def") + 1]

    def use_unixcoder(self, code_string, function_name):
        code = code_string.replace(function_name, "<mask0>")
        print(code)
        tokens_ids = self.model.tokenize([code], max_length=350, mode="<encoder-decoder>")
        source_ids = torch.tensor(tokens_ids).to(self.device)
        prediction_ids = self.model.generate(source_ids, decoder_only=False, beam_size=5, max_length=32)
        predictions = self.model.decode(prediction_ids)
        result = [x.replace("<mask0>", "").strip() for x in predictions[0]]
        print(result)
        return result[0]

    def func_name_gen(self, code_string):
        root = self.parse_code(code_string)
        original_code = self.tokenizer_function(code_string, root)
        print(original_code)
        function_name = self.extract_func_name(original_code)
        print(function_name)
        new_name = self.use_unixcoder(code_string, function_name)
        modified_code_string = " ".join(original_code)
        modified_code_string = modified_code_string.replace(function_name, new_name)
        if modified_code_string.split() != original_code:
            modified_root = self.parse_code(modified_code_string)
            return modified_root, modified_code_string, True
        else:
            return root, code_string, False

    def transform_code(
            self,
            code: Union[str, bytes]
    ) -> Tuple[str, object]:
        root, code, success = self.func_name_gen(code)
        code = re.sub("[ \n\t]+", " ", code)
        return code, {
            "success": success
        }


if __name__ == '__main__':
    java_code = """
    void foo(int n){
        int time = 20;
        if (time < 18) {
          time=10;
        }
         else {
          System.out.println("Good evening.");
        }
    }
    """
    python_code = """def foo(n):
    res = 0
    for i in range(0, 19, 2):
        res += i
    i = 0
    while i in range(n):
        res += i
        i += 1
    return res
    """

    input_map = {
        "java": ("java", java_code),
        "python": ("python", python_code)
    }
    parser_path = 'D:\论文代码开源\CodeGen\evaluate\CodeBLEU\parser\my-languages.so'
    for lang in ["java", "python"]:
        lang, code = input_map[lang]
        func_renamer = FuncNameGen(
            parser_path, lang, "D:\pretrained-model\\unixcoder-base"
        )
        print(lang)
        code, meta = func_renamer.transform_code(code)
        print(re.sub("[ \t\n]+", " ", code))
        print(meta)
        print("=" * 150)
