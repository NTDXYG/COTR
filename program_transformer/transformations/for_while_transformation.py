import copy
import os
import re
from typing import Union, Tuple

import numpy as np

from program_transformer.language_processors import (
    JavaAndCPPProcessor,
    PythonProcessor,
)
from program_transformer.transformations import TransformationBase

processor_function = {
    "java": [JavaAndCPPProcessor.for_to_while_random, JavaAndCPPProcessor.while_to_for_random],
    "python": [PythonProcessor.for_to_while_random],
}


class ForWhileTransformer(TransformationBase):
    """
    Change the `for` loops with `while` loops and vice versa.
    """

    def __init__(self, parser_path, language):
        super(ForWhileTransformer, self).__init__(parser_path=parser_path, language=language)
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
            modified_root, modified_code, success = function(code, self)
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
static int findSum(int[] arr, int n, int k)
{
	int ans = arr[n - k - 1] - arr[0];
	for(int i = 1; i <= k; i++)
	{
		ans = Math . min(arr[n - 1 - (k - i)] - arr[i], ans);
	}
	return ans;
}
    """
    python_code = """
    def is_prime ( n ) :
        if n < 2 :
            return False
        for k in range ( 2 , n ) :
            a . bar ( )
            k += 7
        return True
    """

    input_map = {
        "java": ("java", java_code),
        "python": ("python", python_code),
    }
    parser_path = 'D:\论文代码开源\CodeGen\evaluate\CodeBLEU\parser\my-languages.so'
    for lang in ["java"]:
        lang, code = input_map[lang]
        for_while_transformer = ForWhileTransformer(parser_path, lang)
        print(lang, end="\t")
        code, meta = for_while_transformer.transform_code(code)
        if lang == "python":
            code = PythonProcessor.beautify_python_code(code.split())
        print(code)
        print(meta["success"])
        print("=" * 150)