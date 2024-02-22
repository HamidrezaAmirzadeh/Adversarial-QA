# !pip install datasets
# !pip install huggingface_hub

from datasets import load_dataset
from datasets.dataset_dict import DatasetDict
from huggingface_hub import notebook_login



dataset = load_dataset("SajjadAyoubi/persian_qa")

# remove questions which context does not cover
dataset = dataset.filter(lambda example: example["answers"]['text'] != []) 

# shuffle dataset and select 20% of them
shuffled_dataset = dataset.shuffle(seed=42)
small_train_size = int(0.2*len(dataset['train']))
small_test_size = int(0.2*len(dataset['validation']))
small_train = shuffled_dataset['train'].select(list(range(small_train_size)))
small_test = shuffled_dataset['validation'].select(list(range(small_test_size)))

small_dataset = DatasetDict({'train':small_train, 'validation':small_test})

# login to huggingface and push our small persian-QA to it
notebook_login()
small_dataset.push_to_hub("Hamid-reza/small-persian-QA")