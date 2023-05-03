import copy
import os
import re
from typing import Union, Tuple

import numpy as np

from program_transformer.language_processors import (
    JavaAndCPPProcessor,
    PythonProcessor
)
from program_transformer.transformations import TransformationBase

processor_function = {
    "java": [JavaAndCPPProcessor.operand_swap],
    "python": [PythonProcessor.operand_swap]
}


class OperandSwap(TransformationBase):
    """
    Swapping Operand "a>b" becomes "b<a"
    """

    def __init__(self, parser_path, language):
        super(OperandSwap, self).__init__(parser_path=parser_path, language=language)
        self.language = language
        self.transformations = processor_function[language]
        processor_map = {
            "java": self.get_tokens_with_node_type,
            "python": PythonProcessor.get_tokens
        }
        self.final_processor = processor_map[self.language]

    def transform_code(
            self,
            code: Union[str, bytes],
    ) -> Tuple[str, object]:
        success = False
        transform_functions = copy.deepcopy(self.transformations)
        while not success and len(transform_functions) > 0:
            function = np.random.choice(transform_functions)
            transform_functions.remove(function)
            modified_code, success = function(code, self)
            if success:
                code = modified_code
        root_node = self.parse_code(
            code=code
        )
        return_values = self.final_processor(
            code=code.encode(),
            root=root_node
        )
        if isinstance(return_values, tuple):
            tokens, types = return_values
        else:
            tokens, types = return_values, None
        return re.sub("[ \t\n]+", " ", " ".join(tokens)), \
               {
                   "types": types,
                   "success": success
               }


if __name__ == '__main__':
    java_code = """
        void foo(){
            int time = 20;
            if (time < 18) {
              time=10;
            }
             else {
              System.out.println("Good evening.");
            }
        }
        """
    python_code = """
        def AVLnodes ( height ) :
            if ( height == 0 ) :
                return 1
            elif ( height == 1 ) :
                return 2
            return ( 1 + AVLnodes ( height - 1 ) + AVLnodes ( height - 2 ) )
        """
    input_map = {
        "java": ("java", java_code),
        "python": ("python", python_code)
    }
    parser_path = 'D:\论文代码开源\CodeGen\evaluate\CodeBLEU\parser\my-languages.so'
    for lang in ["python"]:
        lang, code = input_map[lang]
        operandswap = OperandSwap(
            parser_path, lang
        )
        # print(lang)
        # print("-" * 150)
        # print(code)
        # print("-" * 150)
        code, meta = operandswap.transform_code(code)
        print(code)
        # print(JavaAndCPPProcessor.beautify_java_code(code))
        print(meta["success"])
        print("=" * 150)
