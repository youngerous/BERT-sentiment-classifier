# BERT Sentiment Classifier

The version of huggingface nlp library in the reference video does not match with my environment, so there are some changes in code with different version.

## Data
IMDB Dataset (in huggingface nlp library)
- Sampled train/test dataset for toy experiments (sampling ratio: 5%)

## How to run
```shell
# Check flags (for Windows)
>>> python main.py --help

# Debugging mode - the length of dataset is set to the batch size. (for Windows)
>>> python main.py -d
```

## Requirements
- PyTorch Lightning==0.8.5
- nlp==0.4.0
- transformers==3.1.0
- tokenizers==0.8.1rc2

## Reference
[PyTorch sentiment classifier from scratch with Huggingface NLP Library - Full Tutorial Youtube Video](https://www.youtube.com/watch?v=G3pOvrKkFuk)
