import os
import pandas as pd
from datasets import Dataset,DatasetDict
from transformers import AutoModelForSequenceClassification,AutoTokenizer
from pathlib import Path
import numpy as np, matplotlib.pyplot as plt
from numpy.random import normal,seed,uniform

# Global variables
trainFileName = 'train.csv'
testFileName = 'test.csv'
folderPath = 'us-patent-phrase-to-phrase-matching'
modelName = 'microsoft/deberta-v3-small'

# Global settings
# turn of parallelism on local machine
os.environ["TOKENIZERS_PARALLELISM"] = "false"

## Utility Inputs
def tokenize_input(x): return tokenizer(x["input"])

## 1. Generate tokens and labels

def generateTokens(folderPath, trainFile, tokenizer):
    # for working with paths in Python, I recommend using `pathlib.Path`
    path = Path(folderPath)

    # set dataframe to the training data
    df = pd.read_csv(path/trainFile)

    # print(df.describe(include='object'))
    # ID = number representing words
    # anchor = pattern/unique value (Source token)
    # target = What word its comparing it to (target token)
    # context = classification/category
    #
    # PRINT RESULT:
    #                    id                       anchor       target context
    # count              36473                        36473        36473   36473
    # unique             36473                          733        29340     106
    # top     37d61fd2272659b1  component composite coating  composition     H01
    # freq                   1                          152           24    2186

    # Update the dataframe to include a input attribute
    df['input'] = 'TEXT1: ' + df.context + '; TEXT2: ' + df.target + '; ANC1: ' + df.anchor

    # print(df.input.head())
    # 0    TEXT1: A47; TEXT2: abatement of pollution; ANC...
    # 1    TEXT1: A47; TEXT2: act of abating; ANC1: abate...
    # 2    TEXT1: A47; TEXT2: active catalyst; ANC1: abat...
    # 3    TEXT1: A47; TEXT2: eliminating process; ANC1: ...
    # 4    TEXT1: A47; TEXT2: forest region; ANC1: abatement

    # Create dataset from dataframe
    ds = Dataset.from_pandas(df)

    # print(ds)
    # Name: input, dtype: object
    # Dataset({
    #    features: ['id', 'anchor', 'target', 'context', 'score', 'input'],
    #    num_rows: 36473
    # })


    #tokenTest = tokz.tokenize("Here is the big thing about 'things', things make me thangy of that thang. Tik thank [code word] is word!")

    # _indicate a new word.
    # print(tokenTest)
    # ['▁Here', '▁is', '▁the', '▁big', '▁thing', '▁about', "▁'", 'things', "'", ',', 
    # '▁things', '▁make', '▁me', '▁than', 'gy', '▁of', '▁that', '▁than', 'g', '.', 
    # '▁Tik', '▁thank', '▁[', 'code', '▁word', ']', '▁is', '▁word', '!']]
    # Numericalization*: Convert each word (or token) into a number.

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
def createValidatedDataset(testFile, tokenizedDs, tokenizer): 
    eval_df = pd.read_csv(folderPath+'/'+testFile)
    des = eval_df.describe()

    print(des)
    #                       id      anchor                         target context
    # count                 36          36                             36      36
    # unique                36          34                             36      29
    # top     4112d61851461f60  el display  inorganic photoconductor drum     G02
    
    ## Run test set
    eval_df['input'] = 'TEXT1: ' + eval_df.context + '; TEXT2: ' + eval_df.target + '; ANC1: ' + eval_df.anchor
    eval_ds = Dataset.from_pandas(eval_df).map(tokenize_input, batched=True)

    print(eval_ds)
    return eval_ds

# 3. Train model
def train_model(tokenizer, modelName, eval_ds):
    from transformers import TrainingArguments,Trainer
    # get train/test split
    dds = tokenizedDs.train_test_split(0.25, seed=42)

    # Batch size to fit GPU
    bs = 128
    
    # Number
    epochs = 4

    # Learning rate
    lr = 8e-5

    # turn off fp16 due to no CUDA on local machine (mac pro M1)
    has_CUDA = False


    def corr_d(eval_pred): return {'pearson': preds(*eval_pred)}

    args = TrainingArguments('outputs', learning_rate=lr, warmup_ratio=0.1, lr_scheduler_type='cosine', fp16=has_CUDA,
        evaluation_strategy="epoch", per_device_train_batch_size=bs, per_device_eval_batch_size=bs*2,
        num_train_epochs=epochs, weight_decay=0.01, report_to='none')

    model = AutoModelForSequenceClassification.from_pretrained(modelName, num_labels=1)
    trainer = Trainer(model, args, train_dataset=dds['train'], eval_dataset=dds['test'],
    tokenizer=tokenizer, compute_metrics=corr_d)

    trainer.train()
    preds = trainer.predict(eval_ds).predictions.astype(float)
    
    print(preds)
    return preds


## Run
# Tokenization*: Split each text up into words (or actually, as we'll see, into *tokens*)

# `AutoTokenizer` will create a tokenizer appropriate for a given model:

tokenizer = AutoTokenizer.from_pretrained(modelName)
tokenizedDs = generateTokens(folderPath, trainFileName, tokenizer)
eval_ds = createValidatedDataset(testFileName, tokenizedDs, tokenizer)

preds = train_model(tokenizer, modelName, eval_ds)

#show_correlation()


## Process data

## Run Training Process

## Compare different models

## Compare training methods

## Run Validation test
## Run Validation - subset test
## Run Validation - Accurate test

## VISUALIZE UTILITIES

def plot_ds():

    def f(x): return -3*x**2 + 2*x + 20
    def plot_function(f, min=-2.1, max=2.1, color='r'):
        x = np.linspace(min,max, 100)[:,None]
        plt.plot(x, f(x), color)
        def noise(x, scale): return normal(scale=scale, size=x.shape)
        def add_noise(x, mult, add): return x * (1+noise(x,mult)) + noise(x,add)

    def visualizeData():
        np.random.seed(42)
        x = np.linspace(-2, 2, num=20)[:,None]
        y = add_noise(f(x), 0.2, 1.3)
        plt.scatter(x,y)

        plot_function(f, color='b')

    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import make_pipeline

    def plot_poly(degree):
        x = np.linspace(-2, 2, num=20)[:,None]
        y = add_noise(f(x), 0.2, 1.3)
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        model.fit(x, y)
        plt.scatter(x,y)
        plot_function(model.predict)
    
    visualizeData()
    plot_poly(10)
    plt.show()

def show_correlation():
    from sklearn.datasets import fetch_california_housing
    housing = fetch_california_housing(as_frame=True)
    housing = housing['data'].join(housing['target']).sample(1000, random_state=52)
    
    print(housing.head())

    # Show all coefficient correlations for all combinations of columns
    np.set_printoptions(precision=2, suppress=True)
    np.corrcoef(housing, rowvar=False)

    np.corrcoef(housing.MedInc, housing.MedHouseVal)
    # return single coefficient to make splotting easier
    def corr(x,y): return np.corrcoef(x,y)[0][1]

    corr(housing.MedInc, housing.MedHouseVal)

    def show_corr(df, a, b):
        x,y = df[a],df[b]
        plt.scatter(x,y, alpha=0.5, s=4)
        plt.title(f'{a} vs {b}; r: {corr(x, y):.2f}')

    subset = housing[housing.AveRooms<15]
    #show_corr(subset, 'MedInc', 'AveRooms')
    #show_corr(housing, 'MedInc', 'MedHouseVal')
    ##show_corr(housing, 'Population', 'AveRooms')
    show_corr(subset, 'HouseAge', 'AveRooms')
    plt.show()
