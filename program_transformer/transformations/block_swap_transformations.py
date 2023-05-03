import copy
import os
import re
from typing import Union, Tuple

import numpy as np

from program_transformer.language_processors import (
    JavaAndCPPProcessor,
    PythonProcessor,
)

from program_transformer.transformations.transformation_base import TransformationBase

processor_function = {
    "java": [JavaAndCPPProcessor.block_swap_java],
    "python": [PythonProcessor.block_swap],
}


class BlockSwap(TransformationBase):
    """
    Swapping if_else block
    """

    def __init__(self, parser_path, language):
        super(BlockSwap, self).__init__(parser_path=parser_path, language=language)
        self.language = language
        self.transformations = processor_function[language]
        processor_map = {
            "java": self.get_tokens_with_node_type,
            "python": PythonProcessor.get_tokens,
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
    static boolean isWordPresent ( String sentence , String word ) {
      String [ ] s = sentence . split ( " " ) ;
      for ( String temp : s ) {
        if ( temp . compareTo ( word ) == 0 ) {
          return true ;
        } else {
            continue;
        };
      }
      return false ;
    }
    """
    python_code = """
    from typing import List
    
    def factorize(n: int) -> List[int]:
        import math
        fact = []
        i = 2
        while i <= int(math.sqrt(n) + 1):
            if n % i == 0:
                fact.append(i)
                n //= i
            else:
                i += 1
        if n > 1:
            fact.append(n)
        return fact
    """

    input_map = {
        "java": ("java", java_code),
        "python": ("python", python_code)
    }
    # code_directory = os.path.realpath(os.path.join(os.path.realpath(__file__), '../../../../'))
    # parser_path = os.path.join(code_directory, "parser/languages.so")
    for lang in ["python"]:
        lang, code = input_map[lang]
        no_transform = BlockSwap(
            '/home/yangguang/PycharmProjects/RobustCodeTrans/evaluation/CodeBLEU/parser/my-languages.so', lang
        )
        code, meta = no_transform.transform_code(code)
        print(code)
        code = re.sub("[ \t\n]+", " ", code)
        if lang == "python":
            code = PythonProcessor.beautify_python_code(code.split())
        print(code)
        # print(re.sub("[ \t\n]+", " ", code))
        print(meta['success'])

        print("=" * 150)
