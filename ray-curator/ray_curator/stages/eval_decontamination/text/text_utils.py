# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import string
from collections.abc import Callable


def get_word_splitter(language: str) -> Callable[[str], list[str]]:
    """
    For Chinese and Japanese text, we use external libraries to split the text
    because these languages are not separated by spaces. For all other languages,
    such as English, we assume words are separated by spaces.

    Args:
        language (str): An ISO 639-1 language code.
            For example, "en" for English, "zh" for Chinese, and "ja" for Japanese.
    Returns:
        A function which can be used to parse the words of a string into a list.
    """
    language = language.lower()

    if language == "zh":
        # We use the Jieba library which is a Chinese word segmentation module
        # because Chinese text is not separated by spaces.
        import jieba

        def jieba_splitter(text: str) -> list[str]:
            return list(jieba.cut(text))

        return jieba_splitter

    elif language == "ja":
        # We use the MeCab library which is a morphological analyzer for Japanese text
        # because Japanese text is not separated by spaces.
        import MeCab

        def mecab_splitter(text: str) -> list[str]:
            mecab = MeCab.Tagger()
            parsed = mecab.parse(text)
            lines = parsed.strip().split("\n")
            return [line.split("\t")[0] for line in lines if line and line != "EOS"]

        return mecab_splitter

    else:

        def default_splitter(text: str) -> list[str]:
            return text.split()

        return default_splitter


def remove_punctuation(str_in: str) -> str:
    """Remove punctuation from a string."""
    return str_in.translate(str_in.maketrans("", "", string.punctuation))


def get_words(text: str) -> tuple[list[str], list[int]]:
    """
    Extract words from text and return their positions.
    
    Args:
        text (str): Input text to process
        
    Returns:
        tuple: (list of words, list of word start positions)
    """
    word_start_char_positions = []
    prev = 0
    words = []

    text = text.lower()
    text = remove_punctuation(text)
    if len(text) > 0:
        for i in range(len(text)):
            if text[i] != " " and (i == 0 or text[i - 1] == " "):
                word_start_char_positions.append(i)
                if i != 0:
                    words.append(text[prev:i].strip())
                prev = i
        words.append(text[prev : i + 1].strip())
        if words and words[0] == "":
            words = words[1:]
    return words, word_start_char_positions