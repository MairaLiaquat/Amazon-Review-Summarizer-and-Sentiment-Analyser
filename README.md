# Project Overview

This project involves preparing, analyzing, and fine-tuning dataset `McAuley-Lab/Amazon-Reviews-2023` for product reviews. (The finetuned model can be downloded from this [link](https://drive.google.com/drive/folders/1SQZ1QVrkETCadQtaSjWafvBbBTXoQI6j?usp=sharing)) 
It includes data preparation, model fine-tuning, user interaction for review summarization, and sentiment analysis.  
Below are the detailed steps and functionalities:

## Data Preparation

### Download the Dataset

Run the `data_save.py` script to download the dataset.<br>
Match the product names in the metadata using the ASIN code.<br>
Remove any unnecessary details to clean the data.

## Model Fine-Tuning

The `train.py` file is used to fine-tune the BART-large-cnn model for generating summaries of product reviews.

## User Interaction

### Select Product and Get Summarized Reviews

Use the `user_input.ipynb` notebook to:  
- Select a product from a provided list.  
- Obtain a summarized review of the selected product.  
- View the sentiment analysis of the selected product.

## Sentiment Analysis Model Comparison

The `Comapring_Roberta_and_Vader_models.ipynb` notebook is used to compare two models for sentiment analysis:<br>
Here, the comparison is done for a subset of first 250 data samples from the `McAuley-Lab/Amazon-Reviews-2023` dataset.

- **RoBERTa Model:** RoBERTa (Robustly optimized BERT approach) is an advanced transformer-based model designed for natural language understanding. It builds upon the BERT (Bidirectional Encoder Representations from Transformers) architecture
- **VADER Model:** VADER (Valence Aware Dictionary and sEntiment Reasoner) is a rule-based model specifically attuned to sentiment analysis, particularly suited for social media texts. It is designed to be fast and computationally efficient, making it ideal for real-time applications

Both RoBERTa and VADER have their strengths and are suited for different scenarios. RoBERTa, with its transformer-based architecture, excels in deep contextual understanding and complex text analysis. VADER, with its rule-based approach, provides fast and interpretable results but it has limitations to capture wider range of sentiments. 
