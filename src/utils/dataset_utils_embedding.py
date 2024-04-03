import datasets
import random
random.seed(42)
import json
from api_utils import *
import time
import sys
import os 
sys.path.append('..')
# from gensim import corpora
# from gensim.models import TfidfModel
# from gensim.similarities import SparseMatrixSimilarity
# from rank_bm25 import BM25Okapi
import pickle
years = [2011,2012,2013,2014,2015,2016,2017, 2018, 2019, 2020]
# from sentence_transformers import SentenceTransformer, util
# model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
import torch

class medmcqaProcessor:
    def __init__(self, func=None):
        self.template_ner = '''Extract all the biomedicine-related entity from the following question and choices, output each entity in a single line with a serial number (1., 2., ...)
Question: {}
The extracted entities are:
'''
        self.template = '''Question: {} 
Answer: The option is: '''
        self.template_CoT = '''Answering the following question based on the given corpus.
Question: {}
Corpus: {}
Answer: Let's think step by step. '''
        self.template_inference = '''Answering the following question based on the given corpus.
Question: {}
Corpus: {}
Answer: Let's think step by step. {} Therefore, the letter option (only the letter) is:'''
        self.data = json.load(open('../qa/medmcqa_filter.json'))
        self.num2answer = {
            0: 'A',
            1: 'B',
            2: 'C',
            3: 'D'
        }
        self.func = func

        # with open('document_embeddings_sentence.pkl','rb') as f2:
        #     self.document_embeddings = pickle.load(f2)

    # def read_corpus(self):
    #     corpus, document = [], []
    #     for year in years:
    #         with open(os.path.join('..', 'by_year', '{}.pubtator'.format(year))) as f:
    #             for line in f.readlines():
    #                 line = line.strip()
    #                 if '|a|' in line:
    #                     corpus.append(line.split('|a|')[1].lower().split())
    #                     document.append(line.split('|a|')[1])
    #     return corpus, document

    def load_dataset(self):
        return self.data

    def generate_prompt(self, item):
        question = item['question']
        A, B, C, D = item['opa'], item['opb'], item['opc'], item['opd']
        option = '\n'+'A.'+A+'\n'+'B.'+B+'\n'+'C.'+C+'\n'+'D.'+D+'\n'
        question += option

        prompt = self.template.format(question)
        return prompt

    # def generate_prompt(self, item):
    #     question = item['question']
    #     A, B, C, D = item['opa'], item['opb'], item['opc'], item['opd']
    #     option = '\n'+'A.'+A+'\n'+'B.'+B+'\n'+'C.'+C+'\n'+'D.'+D+'\n'
    #     question += option

    #     question_embedding = model.encode(question)
    #     cos_sim = util.cos_sim(question_embedding, self.document_embeddings["embeddings"])
    #     index = torch.argmax(cos_sim[0])
    #     document = self.document_embeddings["document"][index]


    #     prompt_CoT = self.template_CoT.format(question, document)
    #     try:
    #         CoT = self.func(prompt_CoT)
    #     except Exception as E:
    #         print(E)
    #         return None
        
    #     prompt_inference = self.template_inference.format(question, document, CoT)

    #     return prompt_inference

    def parse(self, ret, item):
        ret = ret.replace('.', '')
        if len(ret) > 1:
            ret = ret[0]
        item['PaLM prediction'] = ret
        answer = item['cop']
        answer = self.num2answer[answer]
        if answer.strip() == ret.strip():
            acc = 1
        else:
            acc = 0
        return item, acc


class medqaProcessor:
    def __init__(self, filter=False, func=None):
        self.template_ner = '''Extract all the biomedicine-related entity from the following question and choices, output each entity in a single line with a serial number (1., 2., ...)
Question: {}
The extracted entities are:
'''
        self.template = '''Question: {} 
Answer: The option is: '''
        self.template_CoT = '''Answering the following question based on the given corpus.
Question: {}
Corpus: {}
Answer: Let's think step by step. '''
        self.template_inference = '''Answering the following question based on the given corpus.
Question: {}
Corpus: {}
Answer: Let's think step by step. {} Therefore, the letter option (only the letter) is:'''

        self.data = json.load(open('../qa/medqa_filter.json'))
        self.num2answer = {
            0: 'A',
            1: 'B',
            2: 'C',
            3: 'D'
        }

        self.func = func

        # with open('document_embeddings_sentence.pkl','rb') as f2:
        #     self.document_embeddings = pickle.load(f2)

    # def read_corpus(self):
    #     corpus, document = [], []
    #     for year in years:
    #         with open(os.path.join('..', 'by_year', '{}.pubtator'.format(year))) as f:
    #             for line in f.readlines():
    #                 line = line.strip()
    #                 if '|a|' in line:
    #                     corpus.append(line.split('|a|')[1].lower().split())
    #                     document.append(line.split('|a|')[1])
    #     return corpus, document

    def load_dataset(self):
        return self.data

    def generate_prompt(self, item):
        question = item['question']
        A, B, C, D = item['choices'][0], item['choices'][1], item['choices'][2], item['choices'][3]
        option = '\n'+'A.'+A+'\n'+'B.'+B+'\n'+'C.'+C+'\n'+'D.'+D+'\n'
        question += option

        prompt = self.template.format(question)
        return prompt

    def parse(self, ret, item):
        ret = ret.replace('.', '')
        if len(ret) > 1:
            ret = ret[0]
        item['PaLM prediction'] = ret
        answer = item['answer'][0]
        answer = item['choices'].index(answer)
        answer = self.num2answer[answer]
        if answer.strip() == ret.strip():
            acc = 1
        else:
            acc = 0
        return item, acc


class mmluProcessor:
    def __init__(self, func):
        self.template_ner = '''Extract all the medicine-related entity from the following question and choices, output each entity in a single line with a serial number (1., 2., ...)
Question: {}
The extracted entities are:
'''
        self.template = '''Question: {} 
Answer: The option is: '''
        self.template_CoT = '''Answering the following question based on the given corpus.
Question: {}
Corpus: {}
Answer: Let's think step by step. '''
        self.template_inference = '''Answering the following question based on the given corpus.
Question: {}
Corpus: {}
Answer: Let's think step by step. {} Therefore, the letter option (only the letter) is:'''
        self.data = json.load(open('../qa/mmlu_filter.json'))
        self.num2answer = {
            0: 'A',
            1: 'B',
            2: 'C',
            3: 'D'
        }

        self.func=func

    def read_corpus(self):
        corpus, document = [], []
        for year in years:
            with open(os.path.join('..', 'by_year', '{}.pubtator'.format(year))) as f:
                for line in f.readlines():
                    line = line.strip()
                    if '|a|' in line:
                        corpus.append(line.split('|a|')[1].lower().split())
                        document.append(line.split('|a|')[1])
        return corpus, document

    def load_dataset(self):
        return self.data

    def generate_prompt(self, item):
        question = item['question']
        A, B, C, D = item['choices'][0], item['choices'][1], item['choices'][2], item['choices'][3]
        option = '\n'+'A.'+A+'\n'+'B.'+B+'\n'+'C.'+C+'\n'+'D.'+D+'\n'
        question += option

        prompt = self.template.format(question)
        return prompt

    def parse(self, ret, item):
        ret = ret.replace('.', '')
        if len(ret) > 1:
            ret = ret[0]
        item['PaLM prediction'] = ret
        answer = item['answer']
        # answer = item['choices'].index(answer)
        answer = self.num2answer[answer]
        # print(ret, answer)
        if answer.strip() == ret.strip():
            acc = 1
        else:
            acc = 0
        return item, acc


class qa4mreProcessor:
    def __init__(self, func):
        self.template_ner = '''Extract all the biomedicine-related entity from the following question and choices, output each entity in a single line with a serial number (1., 2., ...)
Question: {}
The extracted entities are:
'''
        self.template = '''Question: \"{}\"
Answer: The option is: '''
        self.template_CoT = '''Answering the following question based on the given corpus.
Question: {}
Corpus: {}
Answer: Let's think step by step. '''
        self.template_inference = '''Answering the following question based on the given corpus.
Question: {}
Corpus: {}
Answer: Let's think step by step. {} Therefore, the letter option (only the letter) is:'''

        self.data = json.load(open('../qa/qa4mre_filter.json'))
        self.num2answer = {
            1: 'A',
            2: 'B',
            3: 'C',
            4: 'D',
            5: 'E'
        }
        self.func=func

    def read_corpus(self):
        corpus, document = [], []
        for year in years:
            with open(os.path.join('..', 'by_year', '{}.pubtator'.format(year))) as f:
                for line in f.readlines():
                    line = line.strip()
                    if '|a|' in line:
                        corpus.append(line.split('|a|')[1].lower().split())
                        document.append(line.split('|a|')[1])
        return corpus, document

    def load_dataset(self):
        return self.data

    def generate_prompt(self, item):
        question = item['question_str']
        A, B, C, D, E = item['answer_options']['answer_str'][0], item['answer_options']['answer_str'][1], item['answer_options']['answer_str'][2], item['answer_options']['answer_str'][3], item['answer_options']['answer_str'][4]
        option = '\n'+'A.'+A+'\n'+'B.'+B+'\n'+'C.'+C+'\n'+'D.'+D+'\n'+'E.'+E+'\n'
        question += option

        prompt = self.template.format(question)
        return prompt

    def parse(self, ret, item):
        ret = ret.replace('.', '')
        if len(ret) > 1:
            ret = ret[0]
        item['PaLM prediction'] = ret
        answer = item['correct_answer_id']
        answer = self.num2answer[int(answer)]
        if answer.strip() == ret.strip():
            acc = 1
        else:
            acc = 0
        return item, acc