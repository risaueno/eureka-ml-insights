from dataclasses import dataclass

import pandas as pd

from .transform import DFTransformBase


@dataclass
class CRUXEvalGenerateQuestion(DFTransformBase):
    """
    This class generates questions for CRUXEval.
    """

    model_code_column: str
    model_input_column: str
    model_question_column: str

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.model_question_column] = df.apply(
            lambda x: self.question_from_code_and_input(x[self.model_code_column], x[self.model_input_column]), axis=1
        )
        return df

    @staticmethod
    def question_from_code_and_input(code: str, input: str) -> str:
        question = code + "\n \nassert f(" + input + ") == ??"
        return question


# @dataclass
# class CRUXEvalExtractAnswer(DFTransformBase):
#     model_output_column: str
#     model_answer_column: str

#     def transform(self, df: pd.DataFrame) -> pd.DataFrame:
#         df[self.model_answer_column] = df[self.model_output_column].apply(self.parse_reasoning_output_answer)
#         return df

#     @staticmethod
#     def parse_reasoning_output_answer(response: str) -> float:
#         """
#         Parse the input string to extract answer of a given CRUXEval question.
#         Parameters:
#             response (str): Input string containing answer X
#         Returns:
#             numerical_value (float): A numeric value representing the model's answer.
#         """

#         if "####" not in response:
#             return None

#         answer = str(response).split("####")[-1].strip()

#         return answer
