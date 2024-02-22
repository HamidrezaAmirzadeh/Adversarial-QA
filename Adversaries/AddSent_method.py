# !pip install transformers[sentencepiece]
# !pip install datasets
# !pip install googletrans==3.1.0a0

import numpy as np
import pandas as pd
from transformers import AutoModelWithLMHead, AutoTokenizer
from transformers import pipeline
from googletrans import Translator


df = pd.read_csv('drive/MyDrive/per_org_loc_dat.csv')
PERSONS = df['PER']
LOCATIONS = df['LOC']
ORGANIZATIONS = df['ORG']



hub_model_id = "Hamid-reza/bert-base-parsbert-uncased-finetuned-conditioned-khorshid"
filler = pipeline(model=hub_model_id)

hub_model_id = "HooshvareLab/bert-base-parsbert-ner-uncased"
ner = pipeline(model=hub_model_id)

hub_model_id = "wietsedv/xlm-roberta-base-ft-udpos28-fa"
pos = pipeline(model=hub_model_id)


tokenizer = AutoTokenizer.from_pretrained("Farnazgh/QA2D")
model = AutoModelWithLMHead.from_pretrained("Farnazgh/QA2D")

def transform_qa2d(question, answer, max_length=300):
  
  text = "q: "+question+" a: "+answer
  input_ids = tokenizer.encode(text, return_tensors="pt", add_special_tokens=True)
  generated_ids = model.generate(input_ids=input_ids, num_beams=2, max_length=max_length, early_stopping=True)[0]
  preds = tokenizer.decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

  return preds


translator_2E = Translator()

def AddSent(question, answer):

  # replace one of NERs of the question
  nameEnts = ner(question)
  ne, start, end = None, 0, 0
  nerTaggs_list = []
  for nameEnt in nameEnts:
    if nameEnt['entity'][0] == 'B':
      nerTaggs_list.append((ne,start,end))
      ne = nameEnt['entity'][2:]
      start = nameEnt['start']
      end = nameEnt['end']
    else:
      end = nameEnt['end']
  nerTaggs_list.append((ne,start,end))
  del nerTaggs_list[0]

  if len(nerTaggs_list) > 0:
    mask_spcs = nerTaggs_list[np.random.randint(len(nerTaggs_list))]
    msk_ne = mask_spcs[0]
    if msk_ne == 'organization':
      random_org = ORGANIZATIONS[np.random.randint(len(ORGANIZATIONS))]
      question = question.replace(question[mask_spcs[1]:mask_spcs[2]], random_org, 1)
    elif msk_ne == 'location':
      random_loc = LOCATIONS[np.random.randint(len(LOCATIONS))]
      question = question.replace(question[mask_spcs[1]:mask_spcs[2]], random_loc, 1)
    elif msk_ne == 'person':
      random_per = PERSONS[np.random.randint(len(PERSONS))]
      question = question.replace(question[mask_spcs[1]:mask_spcs[2]], random_per, 1)
    else:
      question_mask = question.replace(question[mask_spcs[1]:mask_spcs[2]], '[MASK]', 1)
      question = filler(question_mask)[np.random.randint(5)]['sequence']

  # replace one of POS of the question
  posTaggs = pos(question)
  posTaggs_list = []
  for posTagg in posTaggs:
    if posTagg['entity'] in ['ADJ', 'NOUN']:
      start = posTagg['start']
      end = posTagg['end']
      posTaggs_list.append((start,end))

  if len(posTaggs_list) > 0:
    mask_idx = posTaggs_list[np.random.randint(len(posTaggs_list))]
    question_mask = question.replace(question[mask_idx[0]:mask_idx[1]], '[MASK]', 1)
    question = filler(question_mask)[np.random.randint(5)]['sequence']


  # generate fake answer using real answer
  posTaggs = pos(answer)
  posTaggs_list = []
  for posTagg in posTaggs:
    if posTagg['entity'] in ['ADJ', 'NOUN', 'PROPN']:
      start = posTagg['start']
      end = posTagg['end']
      posTaggs_list.append((start,end))

  if len(posTaggs_list) > 0:
    mask_idx = posTaggs_list[np.random.randint(len(posTaggs_list))]
    answer_mask = answer.replace(answer[mask_idx[0]:mask_idx[1]], '[MASK]', 1)
    answer = filler(answer_mask)[np.random.randint(5)]['sequence']

  # convert aggreagtive to declaritive sentence using fake question and fake answer
  question_en = translator_2E.translate(question, dest='en').text
  answer_en = translator_2E.translate(answer, dest='en').text
  result_en = transform_qa2d(question_en, answer_en)
  result = translator_2E.translate(result_en, dest='fa').text
  return result