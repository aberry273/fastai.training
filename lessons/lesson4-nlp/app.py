import os
import pandas as pd
import numpy as np

from datasets import Dataset,DatasetDict
from transformers import AutoModelForSequenceClassification,AutoTokenizer
from transformers import TrainingArguments,Trainer
from pathlib import Path
from numpy.random import normal,seed,uniform

# Global variables
training_data_file_name = 'train.csv'
test_data_file_name = 'test.csv'
folderPath = 'us-patent-phrase-to-phrase-matching'
modelName = 'microsoft/deberta-v3-small'

# Global settings
# turn of parallelism on local machine
parallelism = "true"
os.environ["TOKENIZERS_PARALLELISM"] = parallelism

use_fast_tokenizer = True

# turn off fp16 due to no CUDA on local machine (mac pro M1)
has_CUDA = True

## Utility Inputs
def tokenize_input(x): return tokenizer(x["input"])
def corr(x,y): return np.corrcoef(x,y)[0][1]

## 1. Generate tokens and labels

def createTokenizedDataset(folderPath, trainFile):
    # for working with paths in Python, I recommend using `pathlib.Path`
    path = Path(folderPath)

    # set dataframe to the training data
    df = pd.read_csv(path/trainFile)

    # print(df.describe(include='object'))

    # Update the dataframe to include a input attribute
    df['input'] = 'TEXT1: ' + df.context + '; TEXT2: ' + df.target + '; ANC1: ' + df.anchor

    # print(df.input.head())
    # 4    TEXT1: A47; TEXT2: forest region; ANC1: abatement

    # Create dataset from dataframe
    ds = Dataset.from_pandas(df)

    # print(ds)

    # tokenTest = tokz.tokenize("Here is the big thing about 'things', things make me thangy of that thang. Tik thank [code word] is word!")
    # print(tokenTest)

    # Run the above tokenizer function on all rows (in parallel) in the dataset 
    tok_ds = ds.map(tokenize_input, batched=True)

    # Add an input_ids row
    row = tok_ds[0]
    row['input'], row['input_ids']

    # Use default vocab list of tokenizer, which contains the unique ID for each token in the string
    # ofToken = tokenizer.vocab['▁of']

    # print(ofToken)
    # rename the score to labels as the transformers always assume we have a column called labels for categorization
    tok_ds = tok_ds.rename_columns({'score':'labels'})
    return tok_ds

## 2. Generate tokens and labels
def createValidationDataset(testFile): 
    eval_df = pd.read_csv(folderPath+'/'+testFile)
    
    #des = eval_df.describe()
    #print(des)

    ## Run test set
    eval_df['input'] = 'TEXT1: ' + eval_df.context + '; TEXT2: ' + eval_df.target + '; ANC1: ' + eval_df.anchor
    eval_ds = Dataset.from_pandas(eval_df).map(tokenize_input, batched=True)

    #print(eval_ds)
    return eval_ds

# 3. Train model
def train_model(tokenizer, modelName, tokenized_dataset, eval_ds):
    # get train/test split
    dds = tokenized_dataset.train_test_split(0.25, seed=42)

    # Batch size to fit GPU
    bs = 128
    
    # Number
    epochs = 4

    # Learning rate
    lr = 8e-5

    def corr_d(eval_pred): return {'pearson': corr(*eval_pred)}

    args = TrainingArguments('outputs', learning_rate=lr, warmup_ratio=0.1, lr_scheduler_type='cosine', fp16=has_CUDA,
        evaluation_strategy="epoch", per_device_train_batch_size=bs, per_device_eval_batch_size=bs*2,
        num_train_epochs=epochs, weight_decay=0.01, report_to='none')

    model = AutoModelForSequenceClassification.from_pretrained(modelName, num_labels=1)
    trainer = Trainer(model, args, train_dataset=dds['train'], eval_dataset=dds['test'],
    tokenizer=tokenizer, compute_metrics=corr_d)

    trainer.train()
    pred_results = trainer.predict(eval_ds).predictions.astype(float)
    
    return pred_results


## Run
# Tokenization*: Split each text up into words (or actually, as we'll see, into *tokens*)
# `AutoTokenizer` will create a tokenizer appropriate for a given model:
run_local = False
tokenizer = None

# If not running from huggingface library, use local files in output folder, otherwise load from existing model name
if(run_local == False):
    print('Running Hugginface model')
    tokenizer = AutoTokenizer.from_pretrained(modelName)
else:
    print('Running local model')
    localModelFolder = 'output/checkpoint-500/'
    modelPath = Path(localModelFolder)
    print(modelPath)
    tokenizer = AutoTokenizer.from_pretrained(modelPath, local_files_only=True)

# Create tokensfrom the training file
tokenized_dataset = createTokenizedDataset(folderPath, training_data_file_name)
evaluation_dataset = createValidationDataset(test_data_file_name)

predictions = train_model(tokenizer, modelName, tokenized_dataset, evaluation_dataset)
preds = np.clip(predictions, 0, 1)
print(preds)
#show_correlation()