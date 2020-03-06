## Char-level translation (English to French)
This is a simple Char-level translation (English to French) example in Keras. The data and code was provided by [Keras blog](https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html) or [Keras Example](https://github.com/keras-team/keras/blob/master/examples/lstm_seq2seq.py). 

## Download data
Download [English to French sentence pairs](http://www.manythings.org/anki/fra-eng.zip) and place `fra.txt` in `data/`.
This file is sorted by sentence length. We have an option in the script to shuffle this dataset.

## Usage
```
usage: seq2seq_lstm.py [-h] [--mode {train,infer}] [--batch BATCH_SIZE]
                            [--epochs EPOCHS] [--units NUM_UNITS]
                            [--datapath DATAPATH] [--shuffle_data]
                            [--train_size NUM_TRN_SAMPLES]
                            [--infer_size NUM_INF_SAMPLES]

optional arguments:
  -h, --help            show this help message and exit
  --mode {train,infer}  Either "train" for training or "infer" for inference
  --batch BATCH_SIZE    Batch size
  --epochs EPOCHS       Number of epochs
  --units NUM_UNITS     Number of LSTM units
  --datapath DATAPATH   Path to data [default: fra-eng/fra_shuffled.txt]
  --shuffle_data        Shuffle input data and save
  --train_size NUM_TRN_SAMPLES
                        Number of samples for training
  --infer_size NUM_INF_SAMPLES
                        Number of samples for training
```
