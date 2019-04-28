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
        expected = OrderedDict([('DC.conformsTo', 0.1), ('DC.source', 'Polyglot 16.07.04'), ('DC.created', '2019-01-25T17:04:34'),('DC.date', '2019-01-25T17:04:34'), ('DC.creator', ''), ('DC.publisher', ''), ('DC.title', ''),('DC.description', ''), ('DC.identifier', ''), ('DC.language', 'en'), ('conll', {}), ('documents', [OrderedDict([('text','Autonomous cars from the countryside of France shift insurance liability toward manufacturers. People are afraid that they will crash.'),('tokenList', [{'id': 1, 'text': 'Autonomous', 'characterOffsetBegin': 0, 'characterOffsetEnd': 10,'features': {'Overt': 'Yes'}, 'scores': {'upos': 0, 'xpos': 0, 'entity': 0}, 'misc': {'SpaceAfter': 'Yes'}, 'upos': 'ADJ', 'xpos': 'JJ', 'entity_iob': 'O'}, {'id': 2, 'text': 'cars', 'characterOffsetBegin': 11, 'characterOffsetEnd': 15,'features': {'Overt': 'Yes'}, 'scores': {'upos': 0, 'xpos': 0, 'entity': 0}, 'misc': {'SpaceAfter': 'Yes'}, 'upos': 'NOUN', 'xpos': 'NNS', 'entity_iob': 'O'},  {'id': 3, 'text': 'from', 'characterOffsetBegin': 16, 'characterOffsetEnd': 20, 'features': {'Overt': 'Yes'}, 'scores': {'upos': 0, 'xpos': 0, 'entity': 0}, 'misc': {'SpaceAfter': 'Yes'}, 'upos': 'ADP', 'xpos': 'IN', 'entity_iob': 'O'}, {'id': 4, 'text': 'the', 'characterOffsetBegin': 21, 'characterOffsetEnd': 24,'features': {'Overt': 'Yes'}, 'scores': {'upos': 0, 'xpos': 0, 'entity': 0}, 'misc': {'SpaceAfter': 'Yes'}, 'upos': 'DET', 'xpos': 'DT', 'entity_iob': 'O'},  {'id': 5, 'text': 'countryside', 'characterOffsetBegin': 25, 'characterOffsetEnd': 36, 'features': {'Overt': 'Yes'}, 'scores': {'upos': 0, 'xpos': 0, 'entity': 0}, 'misc': {'SpaceAfter': 'Yes'}, 'upos': 'NOUN', 'xpos': 'NN', 'entity_iob': 'O'}, {'id': 6, 'text': 'of', 'characterOffsetBegin': 37, 'characterOffsetEnd': 39, 'features': {'Overt': 'Yes'}, 'scores': {'upos': 0, 'xpos': 0, 'entity': 0}, 'misc': {'SpaceAfter': 'Yes'}, 'upos': 'ADP', 'xpos': 'IN', 'entity_iob': 'O'},  {'id': 7, 'text': 'France', 'characterOffsetBegin': 40, 'characterOffsetEnd': 46, 'features': {'Overt': 'Yes'}, 'scores': {'upos': 0, 'xpos': 0, 'entity': 0}, 'misc': {'SpaceAfter': 'Yes'}, 'upos': 'PROPN', 'xpos': 'NNP', 'entity': 'S-LOC', 'entity_iob': 'B'}, {'id': 8, 'text': 'shift', 'characterOffsetBegin': 47, 'characterOffsetEnd': 52, 'features': {'Overt': 'Yes'}, 'scores': {'upos': 0, 'xpos': 0, 'entity': 0}, 'misc': {'SpaceAfter': 'Yes'}, 'upos': 'VERB', 'xpos': 'VBP', 'entity_iob': 'O', 'synsets': [{'wordnetId': 'shift.v.01', 'scores': {'wordnetId': 0}}]}, {'id': 9, 'text': 'insurance', 'characterOffsetBegin': 53, 'characterOffsetEnd': 62, 'features': {'Overt': 'Yes'}, 'scores': {'upos': 0, 'xpos': 0, 'entity': 0}, 'misc': {'SpaceAfter': 'Yes'}, 'upos': 'NOUN', 'xpos': 'NN', 'entity_iob': 'O'},{'id': 10, 'text': 'liability', 'characterOffsetBegin': 63, 'characterOffsetEnd': 72, 'features': {'Overt': 'Yes'}, 'scores': {'upos': 0, 'xpos': 0, 'entity': 0}, 'misc': {'SpaceAfter': 'Yes'}, 'upos': 'NOUN', 'xpos': 'NN', 'entity_iob': 'O'}, {'id': 11, 'text': 'toward', 'characterOffsetBegin': 73, 'characterOffsetEnd': 79, 'features': {'Overt': 'Yes'}, 'scores': {'upos': 0, 'xpos': 0, 'entity': 0}, 'misc': {'SpaceAfter': 'Yes'}, 'upos': 'ADP', 'xpos': 'IN', 'entity_iob': 'O'}, {'id': 12, 'text': 'manufacturers.', 'characterOffsetBegin': 80, 'characterOffsetEnd': 94, 'features': {'Overt': 'Yes'}, 'scores': {'upos': 0, 'xpos': 0, 'entity': 0}, 'misc': {'SpaceAfter': 'Yes'}, 'upos': 'NOUN', 'xpos': 'NN', 'entity_iob': 'O'},  {'id': 13, 'text': 'People', 'characterOffsetBegin': 0, 'characterOffsetEnd': 6, 'features': {'Overt': 'Yes'}, 'scores': {'upos': 0, 'xpos': 0, 'entity': 0}, 'misc': {'SpaceAfter': 'Yes'}, 'upos': 'NOUN', 'xpos': 'NNS', 'entity_iob': 'O'}, {'id': 14, 'text': 'are', 'characterOffsetBegin': 7, 'characterOffsetEnd': 10, 'features': {'Overt': 'Yes'}, 'scores': {'upos': 0, 'xpos': 0, 'entity': 0}, 'misc': {'SpaceAfter': 'Yes'}, 'upos': 'AUX', 'xpos': 'VBP', 'entity_iob': 'O', 'synsets': [{'wordnetId': 'be.a.01', 'scores': {'wordnetId': 0}}]}, {'id': 15, 'text': 'afraid', 'characterOffsetBegin': 11, 'characterOffsetEnd': 17, 'features': {'Overt': 'Yes'}, 'scores': {'upos': 0, 'xpos': 0, 'entity': 0}, 'misc': {'SpaceAfter': 'Yes'}, 'upos': 'ADJ', 'xpos': 'JJ', 'entity_iob': 'O'},{'id': 16, 'text': 'that', 'characterOffsetBegin': 18, 'characterOffsetEnd': 22,  'features': {'Overt': 'Yes'}, 'scores': {'upos': 0, 'xpos': 0, 'entity': 0},  'misc': {'SpaceAfter': 'Yes'}, 'upos': 'SCONJ', 'xpos': 'IN', 'entity_iob': 'O'},  {'id': 17, 'text': 'they', 'characterOffsetBegin': 23, 'characterOffsetEnd': 27, 'features': {'Overt': 'Yes'}, 'scores': {'upos': 0, 'xpos': 0, 'entity': 0}, 'misc': {'SpaceAfter': 'Yes'}, 'upos': 'PRON', 'xpos': 'PRP', 'entity_iob': 'O'},  {'id': 18, 'text': 'will', 'characterOffsetBegin': 28, 'characterOffsetEnd': 32, 'features': {'Overt': 'Yes'}, 'scores': {'upos': 0, 'xpos': 0, 'entity': 0}, 'misc': {'SpaceAfter': 'Yes'}, 'upos': 'AUX', 'xpos': 'MD', 'entity_iob': 'O'},  {'id': 19, 'text': 'crash.', 'characterOffsetBegin': 33, 'characterOffsetEnd': 39,  'features': {'Overt': 'Yes'}, 'scores': {'upos': 0, 'xpos': 0, 'entity': 0},  'misc': {'SpaceAfter': 'Yes'}, 'upos': 'VERB', 'xpos': 'VB', 'entity_iob': 'O', 'synsets': [{'wordnetId': 'crash.v.01', 'scores': {'wordnetId': 0}}]}]),  ('clauses', []), ('sentences', [ {'id': '0', 'tokenFrom': 1, 'tokenTo': 13, 'tokens': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 'labels': [{'type': 'sentiment', 'label': 'POSITIVE', 'scores': {'label': 0}}]},  {'id': '1', 'tokenFrom': 13, 'tokenTo': 20, 'tokens': [13, 14, 15, 16, 17, 18, 19], 'labels': [{'type': 'sentiment', 'label': 'POSITIVE', 'scores': {'label': 0}}]}]), ('paragraphs', []), ('dependenciesBasic', []), ('dependenciesEnhanced', []),  ('coreferences', []), ('constituents', []),('expressions', [{'type': 'VP', 'scores': {'type': 0}, 'tokens': [18, 19]}])])])])

        #assert actual == expected, actual
        assert isinstance(actual, OrderedDict)

    # def test_model_not_found(self):
    #     with pytest.raises(ModuleNotFoundError):
    #         get_model('martian_core', False, False)


    def test_validation(self):
        assert validation.is_valid(PolyglotPipeline.process(text, ))
