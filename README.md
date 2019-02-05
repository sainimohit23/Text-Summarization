# Text-Summarization
Text summarization model on tensorflow using seq2seq model.

### Word Embedding
Used [Glove pre-trained vectors](https://nlp.stanford.edu/projects/glove/) to initialize word embedding.


## Requirements
- Python 3
- Tensorflow (>=1.08.0)
- numpy
- pickle
- tqdm
- datetime

## Usage
### Extract Data
Dataset is available at [harvardnlp/sent-summary](https://github.com/harvardnlp/sent-summary). Locate the summary.tar.gz file in project root directory. Then,
```
$ python extractData.py
```

### Prepare Data
```
$ python prepareDataset.py
```
It will save processed data in a pickle file.

### Train model

Run `train.py` to build and train the model.
