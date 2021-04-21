# BERT-Sequence-Labeling
BERT-Sequence-Labeling pytorch implementation


## Requirements
```
Python3
transformers==3.1.0
torch==1.6.1+
argparse==1.4.0
seqeval==0.0.12
```

## Prepare

* Download ``google_model.bin`` from [here](https://drive.google.com/drive/folders/1i67mPV1i2P2IMNTks2PtPeZsDnA8SVQN?usp=sharing), and save it to the ``assets/`` directory.
* Download ``dataset`` from [here](https://drive.google.com/drive/folders/1i67mPV1i2P2IMNTks2PtPeZsDnA8SVQN?usp=sharing), and save it to the ``data/`` directory.

### Model Training

Run example on ATIS dataset.
```
python3 train.py --data_dir data/atis/ --model_path /assets/
```
#### To use your own dataset,  modify the DataProcessor in ``data_utils.py``.

### Model Evaluation

Run example on ATIS dataset.
```
python3 evaluate.py --data_dir data/atis/ --model_path /assets/
```

### Model Prediction

Run example on ATIS dataset.
```
python3 predict.py --data_dir data/atis/ --model_path /assets/
```

### Results


|Dataset        |acc |precsion|recall|f1|
|-------------|------------|------------|---|---|
|ATIS | 98.06      | 95.02     |95.52|95.27|
|Snips| 52.32      | 53.68      |0|0|
