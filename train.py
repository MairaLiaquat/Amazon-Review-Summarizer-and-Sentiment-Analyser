from datasets import load_dataset
from transformers import pipeline
from transformers import AutoModelForSeq2SeqLM,AutoTokenizer,DataCollatorForSeq2Seq
import torch
from datasets import load_from_disk
import evaluate
from transformers import TrainingArguments , Trainer, Seq2SeqTrainingArguments ,Seq2SeqTrainer



#model
model_ckpt= 'facebook/bart-large-cnn'
tokenizer =AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt)

#load the dataset
amazon_train_test_valid_dataset_final= ds = load_from_disk('/home/oviya/NLP_Project/amazon_product_dataset/test')
print(amazon_train_test_valid_dataset_final.features)
print(len(amazon_train_test_valid_dataset_final))
# print(amazon_train_test_valid_dataset_final['train'].features)

#data tokenization
def convert_examples_to_features(example_batch):
    input_encodings = tokenizer(example_batch['text'] , max_length = 512, truncation = True, padding=True )

    with tokenizer.as_target_tokenizer():
        target_encodings = tokenizer(example_batch['title'], max_length = 128, truncation = True ,padding=True)

    return {
        'input_ids' : input_encodings['input_ids'],
        'attention_mask': input_encodings['attention_mask'],
        'labels': target_encodings['input_ids']
    }
amazon_pt = amazon_train_test_valid_dataset_final.map(convert_examples_to_features, batched=True)

# print(amazon_pt['train'].features)
# print(len(amazon_pt['train']))

#traing arguments
data_collator =DataCollatorForSeq2Seq(tokenizer, model=model)

#evaluation

import evaluate
rouge = evaluate.load("rouge")
import numpy as np


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}

training_args=Seq2SeqTrainingArguments(
    output_dir='bart_amazon',
    num_train_epochs=2,
    warmup_steps=500,
    per_device_train_batch_size=12,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    logging_steps=10,
    eval_strategy='epoch',
    eval_steps=500,
    save_steps=1e6,
    gradient_accumulation_steps=16,
    predict_with_generate=True
)

trainer=Seq2SeqTrainer(model=model,args=training_args,tokenizer=tokenizer, data_collator=data_collator,compute_metrics=compute_metrics,train_dataset=amazon_pt['train'])

trainer.train()

trainer.save_model("bart_amazon_model_4")