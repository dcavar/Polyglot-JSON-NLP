#!/usr/bin/env python3

"""
(C) 2019 Damir Cavar, Oren Baldinger, Maanvitha Gongalla, Anurag Kumar, Murali Kammili, Boli Fang

Wrappers for Polyglot to JSON-NLP output format.

Licensed under the Apache License 2.0, see the file LICENSE for more details.

Brought to you by the NLP-Lab.org (https://nlp-lab.org/)!
"""
import functools
from collections import OrderedDict
from typing import Dict, Tuple
import polyglot
import pyjsonnlp
from polyglot.text import Text

from pyjsonnlp.pipeline import Pipeline

name = "polyglotjsonnlp"
__cache = {}
__version__ = "0.2.4"


def cache_it(func):
    """A decorator to cache function response based on params. Add it to top of function as @cache_it."""

    global __cache

    @functools.wraps(func)
    def cached(*args):
        f_name = func.__name__
        s = ''.join(map(str, args))
        if s not in __cache[f_name]:
            __cache[f_name][s] = func(*args)
        return __cache[f_name][s]
    return cached

class PolyglotPipeline(Pipeline):
    @staticmethod
    def get_polyglot_sentences(text, neighbors, d, doc):
        """
        Process a text using polyglot, returning language, named entities, pos tags, morphology, and optionally synonyms
        :param text: The text to process
        :param neighbors: Whether or not to include neighbors

        """
        
        token_id = 1
        token_lookup:  Dict[Tuple[int, int], int] = {}  # map (sent_id, polyglot token index) to our token index
        for sent_num, sent in enumerate(doc.sentences):
            current_sent = {
                'id': str(sent_num),
                'tokenFrom': token_id,
                'tokenTo': token_id + len(sent),  # begin inclusive, end exclusive
                'tokens': []
            }
            #d['sentences'] = current_sent
            #print(current_sent)
            d['id'] = sent_num
            d['text'] = str(sent)
            d['sentences'][current_sent['id']] = current_sent
        
            entities = {}
            for ent in sent.entities:
                for i in range(ent.start, ent.end):
                    entities[i] = ent.tag

            tags = dict((i, tag[1]) for i, tag in enumerate(sent.pos_tags))
            for token_idx, token in enumerate(sent.words):
                token_lookup[(sent_num, token_idx)] = token_id
                t = {
                    'id': token_id,
                    'text': token,
                    'upos': tags[token_idx],
                    'lang': token.language,
                    'morphemes': list(token.morphemes),
                    'labels': [{
                        'type': 'sentiment',
                        'label': str(token.polarity)
                    }],
                    'features': {
                        'Overt': 'Yes'
                    }
                }

                # match wordnet format
                if neighbors:
                    try:
                        s = {'neighbors': [w for w in token.neighbors]}
                        if len(s['neighbors']) > 0:
                            t['synsets'] = [s]
                    except KeyError:
                        pass  # OOV words, e.g. contractions, will throw errors

                # named entities
                if token_idx in entities:
                    t['entity'] = entities[token_idx]  # todo map to common entity types? e.g. No I-LOC, etc.
                    # check if this is the first or an internal token in an entity
                    t['entity_iob'] = 'B' if token_idx-1 not in entities or entities[token_idx] != entities[token_idx-1] else 'I'
                else:
                    t['entity_iob'] = 'O'

                current_sent['tokens'].append(token_id)
                token_id += 1
                #d['tokenList'] = t
                d['tokenList'][t['id']] = t

            # multi-word expressions
            expression_id = 0
            for ent in sent.entities:
                if ent.end - ent.start > 1:
                    d['expressions'].append({
                        'id': expression_id,
                        'type': entities[ent.start],
                        'tokens': [token_lookup[(sent_num, t)] for t in range(ent.start, ent.end)]
                    })
                    expression_id += 1

    @staticmethod
    def get_nlp_json(text, neighbors) -> OrderedDict:
        """Process the Polyglot output into JSON"""

        j: OrderedDict = pyjsonnlp.get_base()
        j['DC.source'] = 'polyglot {}'.format(polyglot.__version__)
        d : OrderedDict = pyjsonnlp.get_base_document(1)
        #j['documents'] = get_base_document(text)
        #d = j.get('documents')[len(j.get('documents'))-1]
        #print(d['id'])
        j['documents'][d['id']] = d
        d['meta']['DC.source'] = 'polyglot {}'.format(polyglot.__version__)
        print(j)
        doc = Text(text)
        d['DC.language'] = doc.language.code

        PolyglotPipeline.get_polyglot_sentences(text, neighbors, d, doc)
        return pyjsonnlp.remove_empty_fields(j)



    @staticmethod
    def process(text: str, neighbors=False, coreferences=False, constituents=False, dependencies=False, expressions=False, **kwargs):
        """Process the text into JSON-NLP
        :param **kwargs:
        """
        return PolyglotPipeline.get_nlp_json(text, neighbors)
    
if __name__ == "__main__":
    test_text = "The Mueller Report is a very long report. We spent a long time analyzing it. Trump wishes we didn't, but that didn't stop the intrepid NlpLab."
    PolyglotPipeline.process(test_text, coreferences=True, constituents=False)
