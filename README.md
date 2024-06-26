# Project Overview
This project involves preparing, analyzing, and fine-tuning datasets for product reviews. It includes data preparation, model fine-tuning, user interaction for review summarization, and sentiment analysis. Below are the detailed steps and functionalities:
## Data Preparation

### Download the Dataset:
Run the data_save.py script to download the dataset.

### Process the Data:
Match the product names in the metadata using the ASIN code.
Remove any unnecessary details to clean the data.

## Model Fine-Tuning
Fine-Tune BART Model:
The train.py file is used to fine-tune the BART-large-cnn model for generating summaries of product reviews.

## User Interaction
Select Product and Get Summarized Reviews:
    Use the user_input.ipynb notebook to:
    Select a product from a provided list.
    Obtain a summarized review of the selected product.
    View the sentiment analysis of the selected product.

## Compare Sentiment Analysis Models:
    
The Comparing_Roberta_and_Vader.ipynb notebook is used to compare two models for sentiment analysis:
RoBERTa Model: A transformer-based model designed for natural language understanding.
VADER Model: A rule-based model specifically attuned to sentiment analysis.

By following these steps and utilizing the provided notebooks, users can interact with the dataset, fine-tune models, and perform comprehensive sentiment analysis on product reviews. If you encounter any issues or need further assistance, please refer to the detailed documentation or contact the project maintainers.
