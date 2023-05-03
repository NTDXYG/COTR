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

processor_function = {
    "java": JavaAndCPPProcessor,
    "python": PythonProcessor,
}

tokenizer_function = {
    "java": get_tokens,
    "python": PythonProcessor.get_tokens,
}


class VarRenamer(TransformationBase):
    def __init__(
            self,
            parser_path: str,
            language: str
    ):
        super(VarRenamer, self).__init__(
            parser_path=parser_path,
            language=language,
        )
        self.language = language
        self.processor = processor_function[self.language]
        self.tokenizer_function = tokenizer_function[self.language]
        # C/CPP: function_declarator
        # Java: class_declaration, method_declaration
        # python: function_definition, call
        # js: function_declaration
        self.not_var_ptype = ["function_declarator", "class_declaration", "method_declaration", "function_definition",
                              "function_declaration", "call", "local_function_statement"]

    def extract_var_names(self, root, code_string):
        var_names = []
        queue = [root]

        while len(queue) > 0:
            current_node = queue[0]
            queue = queue[1:]
            # if (current_node.type == "identifier"):
            #     print("identifier", self.tokenizer_function(code_string, current_node)[0])
            # if (current_node.type == "variable_name"):
            #     print("variable_name", self.tokenizer_function(code_string, current_node)[0])
            if (current_node.type == "identifier" or current_node.type == "variable_name") and str(
                    current_node.parent.type) not in self.not_var_ptype:
                var_names.append(self.tokenizer_function(code_string, current_node)[0])
            for child in current_node.children:
                queue.append(child)
        return var_names

    def var_renaming(self, code_string):
        root = self.parse_code(code_string)
        original_code = self.tokenizer_function(code_string, root)
        # print(" ".join(original_code))
        var_names = self.extract_var_names(root, code_string)
        var_names = list(set(var_names))
        num_to_rename = math.ceil(0.2 * len(var_names))
        random.shuffle(var_names)
        var_names = var_names[:num_to_rename]
        var_map = {}
        for idx, v in enumerate(var_names):
            var_map[v] = f"VAR_{idx}"
        modified_code = []
        for t in original_code:
            if t in var_names:
                modified_code.append(var_map[t])
            else:
                modified_code.append(t)

        modified_code_string = " ".join(modified_code)
        if modified_code != original_code:
            modified_root = self.parse_code(modified_code_string)
            return modified_root, modified_code_string, True
        else:
            return root, code_string, False

    def transform_code(
            self,
            code: Union[str, bytes]
    ) -> Tuple[str, object]:
        root, code, success = self.var_renaming(code)
        code = re.sub("[ \n\t]+", " ", code)
        return code, {
            "success": success
        }


if __name__ == '__main__':
    java_code = """
    class A{
        int foo(int n){
            int res = 0;
            for(int i = 0; i < n; i++) {
                int j = 0;
                while (j < i){
                    res += j; 
                }
            }
            return res;
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
        var_renamer = VarRenamer(
            parser_path, lang
        )
        print(lang)
        code, meta = var_renamer.transform_code(code)
        print(re.sub("[ \t\n]+", " ", code))
        print(meta)
        print("=" * 150)
