# !pip install transformers
# !pip install datasets

from transformers import pipeline
from datasets import load_dataset
import evaluate

org_dataset = load_dataset("Hamid-reza/small-persian-QA")

question_answerer_mbert = pipeline("question-answering", model="arashmarioriyad/mbert_v3", tokenizer="arashmarioriyad/mbert_tokenizer_v3")
question_answerer_parsbert = pipeline("question-answering", model="arashmarioriyad/parsbert_v1", tokenizer="arashmarioriyad/parsbert_tokenizer_v1")

#################### models evaluations on original dataset ####################
evaluate.evaluate_model(question_answerer_mbert, org_dataset['validation'])
evaluate.evaluate_model(question_answerer_parsbert, org_dataset['validation'])