"""Python Typings
Commonly used typings for this project
"""

from typing import Dict, List, Tuple

from torch import Tensor

SentimentEntry = Dict[str, str]
SentimentLexicon = Dict[str, Dict[str, str]]

SplitData = Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor]]
