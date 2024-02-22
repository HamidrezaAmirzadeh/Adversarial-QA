# !pip install huggingface_hub

import sys
sys.path.insert(0, '..')
import Adversaries.AddSent_method
from Adversaries.AddSent_method import AddSent
from datasets import load_dataset
from datasets.dataset_dict import DatasetDict



dataset = load_dataset("Hamid-reza/small-persian-QA")

dataset_train = dataset['train']
dataset_val = dataset['validation']

adversaries_train = []
for data in dataset_train:
  question = data['question']
  answer = data['answers']['text'][0]
  adversary = AddSent(question, answer)
  adversaries_train.append(adversary)

adversaries_val = []
for data in dataset_val:
  question = data['question']
  answer = data['answers']['text'][0]
  adversary = AddSent(question, answer)
  adversaries_val.append(adversary)


def add_adversary(example):
    question = example['question']
    answer = example['answers']['text'][0]
    adversary = next(iter(adversaries))
    new_context = example['context'] + ' ' + adversary
    example["context"] = new_context
    return example

adversaries = iter(adversaries_train)
dataset_train = dataset_train.map(add_adversary)

adversaries = iter(adversaries_val)
dataset_val = dataset_val.map(add_adversary)


adversary_dataset = DatasetDict({'train':dataset_train, 'validation':dataset_val})

# login to huggingface and push our Adv-small-persian-QA to it
notebook_login()
adversary_dataset.push_to_hub("Hamid-reza/Adv-small-persian-QA")
