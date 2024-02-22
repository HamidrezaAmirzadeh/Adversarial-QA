# !pip install transformers
# !pip install evaluate


from transformers import pipeline
import evaluate


metric = evaluate.load("squad")


def evaluate_model(model, dataset):
  answer_refrences = []
  answer_predicts = []
  for example in dataset:
    question = example['question']
    context = example['context']
    predicted_answer = model(question=question, context=context)
    answer_refrence = {"id": str(example["id"]), "answers": example["answers"]}
    answer_refrences.append(answer_refrence)
    answer_predict = {'id':str(example["id"]), 'prediction_text':predicted_answer['answer']}
    answer_predicts.append(answer_predict)

  return metric.compute(predictions=answer_predicts, references=answer_refrences)
