import copy
from typing import Union, Tuple

import numpy as np
import os

from program_transformer.language_processors import (
    JavaAndCPPProcessor,
    PythonProcessor,
)
from program_transformer.transformations import TransformationBase

processor_function = {
    "java": [JavaAndCPPProcessor.incre_decre_removal],
    "python": [PythonProcessor.incre_decre_removal],
}


class ConfusionRemover(TransformationBase):
    """
    Change the `for` loops with `while` loops and vice versa.
    """

    def __init__(self, parser_path, language):
        super(ConfusionRemover, self).__init__(parser_path=parser_path, language=language)
        self.language = language
        if language in processor_function:
            self.transformations = processor_function[language]
        else:
            self.transformations = []
        processor_map = {
            "java": self.get_tokens_with_node_type,  # yes
            "python": PythonProcessor.get_tokens,  # no
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
        return " ".join(tokens), \
            {
                "types": types,
                "success": success
            }


if __name__ == '__main__':
    java_code = """
static Vector < Integer > solve(int X, Vector < Integer > A)
{
    int min = Integer.MAX_VALUE;
    int ind = -1;
    for(int i = 0; i < A.size(); i++)
    {
        if(A.get(i) < min)
        {
            min = A.get(i);
            ind = i;
        }
    }
    int maxIndChosen = X / min;
    Vector < Integer > ans = new Vector < > ();
    if(maxIndChosen == 0)
    {
        return ans;
    }
    for(int i = 0; i < maxIndChosen; i++)
    {
        ans.add(ind);
    }
    int temp = maxIndChosen;
    int sum = maxIndChosen * A.get(ind);
    for(int i = 0; i < ind; i++)
    {
        if(sum - X == 0 || temp == 0) break;
        while((sum - A.get(ind) + A.get(i)) <= X && temp != 0)
        {
            ans.remove(0);
            ans.add(i);
            temp--;
            sum += (A.get(i) - A.get(ind));
        }
    }
    Collections.sort(ans);
    return ans;
} 
"""
    python_code = """
def solve(X, A):
    min = 1000000000
    ind = -1
    while i < len(A):
        if A[i] < min:
            min = A[i]
            ind = i
        i += 1
    maxIndChosen = X // min
    ans = []
    if maxIndChosen == 0:
        return ans
    """
    input_map = {
        "python": ("python", python_code)
    }
    parser_path = 'D:\论文代码开源\CodeGen\evaluate\CodeBLEU\parser\my-languages.so'
    for lang in ["python"]:
        lang, code = input_map[lang]
        confusion_remover = ConfusionRemover(
            parser_path, lang
        )
        code, types = confusion_remover.transform_code(code)
        print(types["success"])
        print(code)
        if lang == "python":
            code = PythonProcessor.beautify_python_code(code.split())
        print(code)
        print("=" * 100)