# -*- coding: utf-8 -*-
"""Back_translation_v2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/18fsNmBfID2GnUA9XAs4K0vflmWzr4k9K
"""

#!pip install googletrans==3.1.0a0
#!pip install datasets
#!pip install huggingface_hub

# import Libs
from googletrans import Translator
import datasets
from datasets import load_dataset

import csv

from tqdm import tqdm

from datasets.dataset_dict import DatasetDict
from huggingface_hub import notebook_login

#Module for Translate to Middle language and destination language
translator_2E = Translator()

translator_2P = Translator()
translation = translator_2P.translate(x, dest='fa')
print(translation.text)

"""##Load Dataset That Hamidreza Edited Himself"""

#dataset that created by hamidreza
naiive_dataset = load_dataset("Hamid-reza/small-persian-QA")

naiive_dataset

naiive_dataset['validation'][1]['answers']['text'][0]

#Middle language : France
def back_translation_fr(question) :
  translator_2E = Translator()
  translator_2Fa = Translator()
  # dest = middle_lang
  # _fr = translator_2E.translate(question['answers']['text'][0], dest='fr')
  _fr = translator_2E.translate(question['question'], dest='fr')
  _fa = translator_2Fa.translate(_fr.text, dest='fa')
  question['question'] = _fa.text
  return question
bt_dataset_fr = naiive_dataset.map(back_translation_fr)

bt_dataset_fr['validation'][1]['answers']['text'][0]

#Middle language : English
def back_translation_en(question) :
  translator_2E = Translator()
  translator_2Fa = Translator()
  # dest = middle_lang
  _en = translator_2E.translate(question['question'], dest='en')
  _fa = translator_2Fa.translate(_en.text, dest='fa')
  question['question'] = _fa.text
  return question
bt_dataset_en = naiive_dataset.map(back_translation_en)

#Middle language : armania
def back_translation_hy(question) :
  translator_2hy = Translator()
  translator_2Fa = Translator()
  # dest = middle_lang
  _hy = translator_2hy.translate(question['question'], dest='hy')
  _fa = translator_2Fa.translate(_hy.text, dest='fa')
  question['question'] = _fa.text
  return question
bt_dataset_hy = naiive_dataset.map(back_translation_hy)

# login to huggingface and push our Adv-small-persian-QA to it
notebook_login()

bt_dataset_fr.push_to_hub("jalalnb/back_translation_fr_on_small_persian_QA")
bt_dataset_en.push_to_hub("jalalnb/back_translation_en_on_small_persian_QA")
bt_dataset_hy.push_to_hub("jalalnb/back_translation_hy_on_small_persian_QA")
