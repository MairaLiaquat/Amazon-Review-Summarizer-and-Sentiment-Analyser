from datasets import load_dataset
from transformers import pipeline
from transformers import AutoModelForSeq2SeqLM,AutoTokenizer
import torch
from datasets import DatasetDict

meta_dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_meta_All_Beauty",trust_remote_code=True)
review_dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_All_Beauty", trust_remote_code=True)

asin_to_title = {item["parent_asin"]: item["title"] for item in meta_dataset["full"]}

train_test_dataset=review_dataset['full'].train_test_split(test_size=0.5)
train_test_valid_dataset = train_test_dataset["test"].train_test_split(test_size=0.5)
train_test_valid_dataset = DatasetDict({
    'train': train_test_dataset['train'],
    'test': train_test_valid_dataset['test'],
    'valid': train_test_valid_dataset['train']})


def add_new_column(example):
    parent_asin = example["parent_asin"]
    try:
        new_value = asin_to_title[parent_asin]
    except KeyError:
        new_value = " "
    example["product_name"] = new_value
    return example

dataset=train_test_valid_dataset['train']
amazon_train = dataset.map(add_new_column)

dataset=train_test_valid_dataset['test']
amazon_test = dataset.map(add_new_column)

dataset=train_test_valid_dataset['valid']
amazon_valid = dataset.map(add_new_column)

amazon_train_test_valid_dataset_final = DatasetDict({
    'train':amazon_train,
    'test': amazon_test,
    'valid':amazon_valid})

amazon_train_test_valid_dataset_final['train']=amazon_train_test_valid_dataset_final['train'].remove_columns(['images','asin','user_id','timestamp','helpful_vote','verified_purchase','parent_asin'])
amazon_train_test_valid_dataset_final['test']=amazon_train_test_valid_dataset_final['test'].remove_columns(['images','asin','user_id','timestamp','helpful_vote','verified_purchase','parent_asin'])
amazon_train_test_valid_dataset_final['valid']=amazon_train_test_valid_dataset_final['valid'].remove_columns(['images','asin','user_id','timestamp','helpful_vote','verified_purchase','parent_asin'])

print(amazon_train_test_valid_dataset_final['train'].features)
print(amazon_train_test_valid_dataset_final['test'].features)
print(amazon_train_test_valid_dataset_final['valid'].features)

amazon_train_test_valid_dataset_final.save_to_disk("amazon_product_dataset")