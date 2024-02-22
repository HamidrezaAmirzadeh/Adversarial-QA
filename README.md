# Adversarial-QA

This is our final project of NLP course instructed by Dr. Asgari at Sharif University Technology


**🚀️Contributers :** Hamidreza Amirzadeh, Mohammad Hossein Sameti, Arash Maryoriad, Jalal Nematbakhsh


In this project we aimed to create ***adversarial examples*** for the persian state-of-the-art **question answering(QA)** models. The webdemo of the project is available at this [link](https://huggingface.co/spaces/arashmarioriyad/nlp_project_webdemo_v1)

🎉️First we selected the [Persian-QA](https://huggingface.co/datasets/SajjadAyoubi/persian_qa) dataset. Since the high amount of time needed to create adversarial examples for the entire dataset, we selected only a subset of it called [Small-Persain-QA](https://huggingface.co/datasets/Hamid-reza/small-persian-QA).

In order to generate adversarial examples, we used some techniques you can observe in the tables below.

We used two persain QA models : mbert, Parsbert

👀️In the table below you can see the f1 acurracy of models on the original and adversarial datasets:


| Dataset       | mbert | parsbert |
| --------------- | ------- | ---------- |
| Original      | 57.9  | 56.1     |
| AddSent       | 50.6  | 53.7     |
| AddAny        | 40.5  | 50.3     |
| BackTrans-hy  | 53.7  | 55.7     |
| BackTrans-en  | 56.4  | 56.6     |
| BackTrans-fr  | 55.4  | 51.1     |
| InvisibleChar | 22.3  | 19.3     |



👀️In the table below you can see the f1 acuracy of models on the original and adversarial datasets *after fine-tuning them on each adversarial datasets*. Note that the right hand side of the numbers is the model acuracy on the original dataset.


| Dataset       | mbert       | parsbert    |
| --------------- | ------------- | ------------- |
| AddSent       | 65.4 / 57.6 | 57.8 / 49.3 |
| AddAny        | 55.3 / 58.3 | 53.2 / 53.7 |
| BackTrans-hy  | 57.2 / 56.4 | 52.1 / 52.4 |
| BackTrans-en  | -           | -           |
| BackTrans-fr  | -           | -           |
| InvisibleChar | 7.3 / 17.9  | 4.1 / 13.5  |
