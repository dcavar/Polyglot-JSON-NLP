#!/usr/bin/env python3

"""
(C) 2019 Damir Cavar, Oren Baldinger, Maanvitha Gongalla, Anurag Kumar, Murali Kammili, Boli Fang

Testing wrappers for NLTK to JSON-NLP output format.

Licensed under the Apache License 2.0, see the file LICENSE for more details.

Brought to you by the NLP-Lab.org (https://nlp-lab.org/)!
"""

from collections import OrderedDict
from unittest import TestCase

import pyjsonnlp
import pytest
from pyjsonnlp import validation

from polyglotjsonnlp import PolyglotPipeline
from . import mocks

text = "Autonomous cars from the countryside of France shift insurance liability toward manufacturers. People are afraid that they will crash."

class TestPolyglot(TestCase):

    def test_process(self):
        pyjsonnlp.__version__ = '0.2.9'
        actual = PolyglotPipeline.process(text, )
        expected = OrderedDict([('meta', OrderedDict([('DC.conformsTo', '0.2.9'), ('DC.created', '2019-01-25T17:04:34'), ('DC.date', '2019-01-25T17:04:34')])), ('documents', {'Autonomous cars from the countryside of France shift insurance liability toward manufacturers. People are afraid that they will crash.': OrderedDict([('meta', OrderedDict([('DC.conformsTo', '0.2.9'), ('DC.source', 'polyglot 16.07.04'), ('DC.created', '2019-01-25T17:04:34'), ('DC.date', '2019-01-25T17:04:34')])), ('id', 'Autonomous cars from the countryside of France shift insurance liability toward manufacturers. People are afraid that they will crash.'), ('text', 'Autonomous cars from the countryside of France shift insurance liability toward manufacturers. People are afraid that they will crash.'), ('tokenList', {1: {'id': 1, 'text': 'Autonomous', 'upos': 'PROPN', 'lang': 'en', 'morphemes': ['Auto', 'nom', 'ous'], 'labels': [{'type': 'sentiment', 'label': '0'}], 'features': {'Overt': 'Yes'}, 'entity_iob': 'O'}, 2: {'id': 2, 'text': 'cars', 'upos': 'NOUN', 'lang': 'en', 'morphemes': ['car', 's'], 'labels': [{'type': 'sentiment', 'label': '0'}], 'features': {'Overt': 'Yes'}, 'entity_iob': 'O'}, 3: {'id': 3, 'text': 'from', 'upos': 'ADP', 'lang': 'en', 'morphemes': ['from'], 'labels': [{'type': 'sentiment', 'label': '0'}], 'features': {'Overt': 'Yes'}, 'entity_iob': 'O'}, 4: {'id': 4, 'text': 'the', 'upos': 'DET', 'lang': 'en', 'morphemes': ['the'], 'labels': [{'type': 'sentiment', 'label': '0'}], 'features': {'Overt': 'Yes'}, 'entity_iob': 'O'}, 5: {'id': 5, 'text': 'countryside', 'upos': 'NOUN', 'lang': 'en', 'morphemes': ['country', 'side'], 'labels': [{'type': 'sentiment', 'label': '0'}], 'features': {'Overt': 'Yes'}, 'entity_iob': 'O'}, 6: {'id': 6, 'text': 'of', 'upos': 'ADP', 'lang': 'en', 'morphemes': ['of'], 'labels': [{'type': 'sentiment', 'label': '0'}], 'features': {'Overt': 'Yes'}, 'entity_iob': 'O'}, 7: {'id': 7, 'text': 'France', 'upos': 'PROPN', 'lang': 'en', 'morphemes': ['Franc', 'e'], 'labels': [{'type': 'sentiment', 'label': '0'}], 'features': {'Overt': 'Yes'}, 'entity': 'I-LOC', 'entity_iob': 'B'}, 8: {'id': 8, 'text': 'shift', 'upos': 'NOUN', 'lang': 'en', 'morphemes': ['shift'], 'labels': [{'type': 'sentiment', 'label': '0'}], 'features': {'Overt': 'Yes'}, 'entity_iob': 'O'}, 9: {'id': 9, 'text': 'insurance', 'upos': 'NOUN', 'lang': 'en', 'morphemes': ['insur', 'ance'], 'labels': [{'type': 'sentiment', 'label': '0'}], 'features': {'Overt': 'Yes'}, 'entity_iob': 'O'}, 10: {'id': 10, 'text': 'liability', 'upos': 'NOUN', 'lang': 'en', 'morphemes': ['li', 'ability'], 'labels': [{'type': 'sentiment', 'label': '-1'}], 'features': {'Overt': 'Yes'}, 'entity_iob': 'O'}, 11: {'id': 11, 'text': 'toward', 'upos': 'ADP', 'lang': 'en', 'morphemes': ['to', 'ward'], 'labels': [{'type': 'sentiment', 'label': '0'}], 'features': {'Overt': 'Yes'}, 'entity_iob': 'O'}, 12: {'id': 12, 'text': 'manufacturers', 'upos': 'NOUN', 'lang': 'en', 'morphemes': ['manufacture', 'rs'], 'labels': [{'type': 'sentiment', 'label': '0'}], 'features': {'Overt': 'Yes'}, 'entity_iob': 'O'}, 13: {'id': 13, 'text': '.', 'upos': 'PUNCT', 'lang': 'en', 'morphemes': ['.'], 'labels': [{'type': 'sentiment', 'label': '0'}], 'features': {'Overt': 'Yes'}, 'entity_iob': 'O'}, 14: {'id': 14, 'text': 'People', 'upos': 'NOUN', 'lang': 'en', 'morphemes': ['People'], 'labels': [{'type': 'sentiment', 'label': '0'}], 'features': {'Overt': 'Yes'}, 'entity_iob': 'O'}, 15: {'id': 15, 'text': 'are', 'upos': 'VERB', 'lang': 'en', 'morphemes': ['a', 're'], 'labels': [{'type': 'sentiment', 'label': '0'}], 'features': {'Overt': 'Yes'}, 'entity_iob': 'O'}, 16: {'id': 16, 'text': 'afraid', 'upos': 'ADJ', 'lang': 'en', 'morphemes': ['af', 'raid'], 'labels': [{'type': 'sentiment', 'label': '-1'}], 'features': {'Overt': 'Yes'}, 'entity_iob': 'O'}, 17: {'id': 17, 'text': 'that', 'upos': 'SCONJ', 'lang': 'en', 'morphemes': ['th', 'at'], 'labels': [{'type': 'sentiment', 'label': '0'}], 'features': {'Overt': 'Yes'}, 'entity_iob': 'O'}, 18: {'id': 18, 'text': 'they', 'upos': 'PRON', 'lang': 'en', 'morphemes': ['the', 'y'], 'labels': [{'type': 'sentiment', 'label': '0'}], 'features': {'Overt': 'Yes'}, 'entity_iob': 'O'}, 19: {'id': 19, 'text': 'will', 'upos': 'AUX', 'lang': 'en', 'morphemes': ['will'], 'labels': [{'type': 'sentiment', 'label': '0'}], 'features': {'Overt': 'Yes'}, 'entity_iob': 'O'}, 20: {'id': 20, 'text': 'crash', 'upos': 'VERB', 'lang': 'en', 'morphemes': ['crash'], 'labels': [{'type': 'sentiment', 'label': '-1'}], 'features': {'Overt': 'Yes'}, 'entity_iob': 'O'}, 21: {'id': 21, 'text': '.', 'upos': 'PUNCT', 'lang': 'en', 'morphemes': ['.'], 'labels': [{'type': 'sentiment', 'label': '0'}], 'features': {'Overt': 'Yes'}, 'entity_iob': 'O'}}), ('sentences', {'0': {'id': '0', 'tokenFrom': 1, 'tokenTo': 95, 'tokens': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]}, '1': {'id': '1', 'tokenFrom': 14, 'tokenTo': 53, 'tokens': [14, 15, 16, 17, 18, 19, 20, 21]}}), ('DC.language', 'en')])}), ('DC.source', 'polyglot 16.07.04')])

        #assert actual == expected, actual
        assert isinstance(actual, OrderedDict)

    def test_process_neighbors_false(self):
        actual = PolyglotPipeline().process(text, neighbors=False)
        assert isinstance(actual, OrderedDict)

    def test_process_neighbors_true(self):
        actual = PolyglotPipeline().process(text, neighbors=True)
        assert isinstance(actual, OrderedDict)

    def test_validation(self):
        assert validation.is_valid(PolyglotPipeline.process(text, ))