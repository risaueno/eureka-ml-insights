# write unit tests for the classes in data_utils/transform.py

import logging
import unittest

import numpy as np
import pandas as pd

from eureka_ml_insights.data_utils.cruxeval_utils import CRUXEvalGenerateQuestion

log = logging.getLogger("CRUXEval_GenerateQuestion_tests")


class TestCRUXEvalGenerateQuestion(unittest.TestCase):
    def setUp(self):
        # Make varied testcases with different number of inputs and data types
        codes = [
            "def f(x):\n    return x + 1",
            "def f(x, y):\n    return x * y",
            "def f(text):\n    return text[::-1]",
            "def f(x):\n    if x > 0:\n        return 'positive'\n    else:\n        return 'negative'",
            "def f(x):\n    a = ['e']\n    a.append(x)\n    return a",
        ]
        inputs = [
            "1",
            "2, 3",
            "'hello'",
            "0",
            "'world'",
        ]
        self.questions = [
            "def f(x):\n    return x + 1\n \nassert f(1) == ??",
            "def f(x, y):\n    return x * y\n \nassert f(2, 3) == ??",
            "def f(text):\n    return text[::-1]\n \nassert f('hello') == ??",
            "def f(x):\n    if x > 0:\n        return 'positive'\n    else:\n        return 'negative'\n \nassert f(0) == ??",
            "def f(x):\n    a = ['e']\n    a.append(x)\n    return a\n \nassert f('world') == ??",
        ]

        self.df = pd.DataFrame(columns=["code", "input", "question"])
        self.df["code"] = codes
        self.df["input"] = inputs


    def test_answerextraction(self):
        transform = CRUXEvalGenerateQuestion("code", "input", "question")
        transform.transform(self.df)
        np.testing.assert_array_equal(self.df["question"].values, self.questions)


if __name__ == "__main__":
    unittest.main()
