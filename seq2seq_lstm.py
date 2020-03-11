"""
Explanation:
https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html

Original Code:
https://github.com/fchollet/keras/blob/master/examples/lstm_seq2seq.py

Modified by: Chaunte Lacewell

Sequence to sequence example in Keras (character-level).

This script demonstrates how to implement a basic character-level
sequence-to-sequence model. We apply it to translating
short English sentences into short French sentences,
character-by-character. Note that it is fairly unusual to
do character-level machine translation, as word-level
models are more common in this domain.

# Summary of the algorithm:

- We start with input sequences from a domain (e.g. English sentences)
    and correspding target sequences from another domain
    (e.g. French sentences).
- An encoder LSTM turns input sequences to 2 state vectors
    (we keep the last LSTM state and discard the outputs).
- A decoder LSTM is trained to turn the target sequences into
    the same sequence but offset by one timestep in the future,
    a training process called "teacher forcing" in this context.
    Is uses as initial state the state vectors from the encoder.
    Effectively, the decoder learns to generate `targets[t+1...]`
    given `targets[...t]`, conditioned on the input sequence.
- In inference mode, when we want to decode unknown input sequences, we:
    - Encode the input sequence into state vectors
    - Start with a target sequence of size 1
        (just the start-of-sequence character)
    - Feed the state vectors and 1-char target sequence
        to the decoder to produce predictions for the next character
    - Sample the next character using these predictions
        (we simply use argmax).
    - Append the sampled character to the target sequence
    - Repeat until we generate the end-of-sequence character or we
        hit the character limit.

# Data download:

English to French sentence pairs.
http://www.manythings.org/anki/fra-eng.zip

Lots of neat sentence pairs datasets can be found at:
http://www.manythings.org/anki/

# References:

- Sequence to Sequence Learning with Neural Networks
    https://arxiv.org/abs/1409.3215
- Learning Phrase Representations using
    RNN Encoder-Decoder for Statistical Machine Translation
    https://arxiv.org/abs/1406.1078
"""
from __future__ import print_function
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from nltk.translate.bleu_score import corpus_bleu
import numpy as np
import os
import argparse
import random
import io
from pathlib import Path
from tools.utils import *


def main(input_params):
    current_mode = input_params.mode
    batch_size = input_params.batch_size  # Batch size for training.
    epochs = input_params.epochs  # Number of epochs to train for.
    num_units = input_params.num_units  # dimensionality of the output space
    datafilename = input_params.datapath
    shuffle_data = input_params.shuffle_data

    # Path to the data txt file on disk.
    # data_path = os.path.realpath(os.getcwd()) + datafilename
    train_percentage = input_params.train_percentage  # Number of samples to train on.
    # train_indices = range(0, num_samples)
    # test_indices = range(num_samples, min(num_samples + num_samples_inf, args.num_lines))
    max_encoder_seq_length, num_encoder_tokens, \
        max_decoder_seq_length, num_decoder_tokens, \
        input_token_index, target_token_index, \
        train_input_texts, train_target_texts, \
        val_input_texts, val_target_texts, \
        test_input_texts, test_target_texts = prepare_data(datafilename, train_percentage, shuffle=shuffle_data)
    
    # Test data
    test_encoder_input_data, test_decoder_input_data, test_decoder_target_data = \
        get_input_and_target_data(train_input_texts, train_target_texts, max_encoder_seq_length, max_decoder_seq_length,
        # get_input_and_target_data(test_input_texts, test_target_texts, max_encoder_seq_length, max_decoder_seq_length,
                                      num_encoder_tokens, num_decoder_tokens, input_token_index, target_token_index)

    if current_mode == 'train':
        print("TRAINING====================================")
         # Train data
        train_encoder_input_data, train_decoder_input_data, train_decoder_target_data = \
            get_input_and_target_data(train_input_texts, train_target_texts, max_encoder_seq_length, max_decoder_seq_length,
                                      num_encoder_tokens, num_decoder_tokens, input_token_index, target_token_index)

        # Val data
        val_encoder_input_data, val_decoder_input_data, val_decoder_target_data = \
            get_input_and_target_data(train_input_texts, train_target_texts, max_encoder_seq_length, max_decoder_seq_length,
            # get_input_and_target_data(val_input_texts, val_target_texts, max_encoder_seq_length, max_decoder_seq_length,
                                      num_encoder_tokens, num_decoder_tokens, input_token_index, target_token_index)
        
        model, encoder_model, decoder_model = define_model(num_encoder_tokens, num_decoder_tokens, num_units)

        # Run training
        early_stop = EarlyStopping(monitor='val_loss', verbose=1)
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
        model.fit([train_encoder_input_data, train_decoder_input_data], train_decoder_target_data,
                  batch_size=batch_size, callbacks=[early_stop],
                  epochs=epochs,
                  validation_data=([val_encoder_input_data, val_decoder_input_data], val_decoder_target_data), shuffle=True)

        # Save all models
        os.makedirs('models', exist_ok=True)
        encoder_model, decoder_model = save_trained_models(model, encoder_model, decoder_model, num_units)
        print("TEST====================================")
        eval_test_set(encoder_model, decoder_model, test_input_texts, test_target_texts, test_encoder_input_data,
                          num_encoder_tokens, num_decoder_tokens,
                          target_token_index, max_decoder_seq_length)

    elif current_mode == 'infer':
        print("TEST====================================")
        encoder_model, decoder_model = load_defined_models(num_units)
        eval_test_set(encoder_model, decoder_model, test_input_texts, test_target_texts, test_encoder_input_data,
                          num_encoder_tokens, num_decoder_tokens,
                          target_token_index, max_decoder_seq_length)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, dest='mode', default='infer',
                        choices=['train', 'infer'],
                        help='Either "train" for training or "infer" for inference')
    parser.add_argument("--batch", type=int, dest='batch_size', default=256,
                        help='Batch size')
    parser.add_argument("--epochs", type=int, dest='epochs', default=100,
                        help='Number of epochs')
    parser.add_argument("--units", type=int, dest='num_units', default=256,
                        help='Number of LSTM units')
    parser.add_argument("--datapath", type=Path, dest='datapath', default='data/fra.txt', #'data/fra_shuffled.txt'
                        help='Path to data [default: data/fra_shuffled.txt]')
    parser.add_argument("--shuffle_data", action='store_true',
                        help="Shuffle input data and save")
    parser.add_argument("--train_perc", type=float, dest='train_percentage', default=0.1,
                        help='Percent of data for training') #val and test are same 
    args = parser.parse_args()

    # Only shuffle data for training
    args.shuffle_data = args.shuffle_data if args.mode == 'train' else False
    args.datapath = os.path.join(os.path.realpath(os.getcwd()), str(args.datapath))
    args.num_lines = sum(1 for line in io.open(args.datapath, 'r', encoding='utf-8'))

    main(args)
