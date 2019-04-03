#!/usr/bin/env python3

"""
(C) 2019 Damir Cavar, Oren Baldinger, Maanvitha Gongalla, Anurag Kumar, Murali Kammili, Boli Fang

Testing wrappers for NLTK to JSON-NLP output format.

Licensed under the Apache License 2.0, see the file LICENSE for more details.

Brought to you by the NLP-Lab.org (https://nlp-lab.org/)!
"""

from unittest import TestCase, mock

import pytest

class TestPolyglot(TestCase):
    @mock.patch('settings.version')
    def test_process(self, version):

        text = "Autonomous cars from the countryside of France shift insurance liability toward manufacturers. People are afraid that they will crash."
        polyglotjsonnlp.process(text)