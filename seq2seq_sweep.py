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
from tools.utils import define_model, get_input_and_target_data, eval_test_set

def get_data(args):
    input_texts = []
    target_texts = []
    input_characters = set()
    target_characters = set()
    with io.open(args.datapath, 'r', encoding='utf-8') as f:
        # data = [line.splitlines()[0] for lidx, line in enumerate(f) if lidx < args.numtrain]
        for lidx, rawline in enumerate(f):
            if lidx < args.numtrain:
                line = rawline.splitlines()[0]
                input_text, target_text = line.split('\t')
                # We use "tab" as the "start sequence" character
                # for the targets, and "\n" as "end sequence" character.
                target_text = '\t' + target_text + '\n'
                input_texts.append(input_text)
                target_texts.append(target_text)
                for char in input_text:
                    if char not in input_characters:
                        input_characters.add(char)
                for char in target_text:
                    if char not in target_characters:
                        target_characters.add(char)
                
    input_characters = sorted(list(input_characters))
    target_characters = sorted(list(target_characters))
    return input_texts, target_texts, input_characters, target_characters
    

def main(input_params):
    input_texts, target_texts, input_characters, target_characters = get_data(input_params) #returns numtrain samples

    input_token_index = dict(
        [(char, i) for i, char in enumerate(input_characters)])
    target_token_index = dict(
        [(char, i) for i, char in enumerate(target_characters)])     
    
    num_encoder_tokens = input_params.enc_dim
    num_decoder_tokens = input_params.dec_dim
    max_encoder_seq_length = max([len(txt) for txt in input_texts]) #Max sentence length
    max_decoder_seq_length = max([len(txt) for txt in target_texts]) #Max sentence length
    
    print('Data path:', input_params.datapath)
    print('Number of samples:', len(input_texts))
    print('Number of unique input tokens:', num_encoder_tokens)
    print('Number of unique output tokens:', num_decoder_tokens)
    print('Max sequence length for inputs:', max_encoder_seq_length)
    print('Max sequence length for outputs:', max_decoder_seq_length)            
    print('Number of training samples:', args.numtrain * .75)
    print('Number of validation samples:', args.numtrain * .25)
    print('Number of testing samples:', args.numtrain)
    
    # Test data
    encoder_input_data, decoder_input_data, decoder_target_data = \
        get_input_and_target_data(input_texts, target_texts, max_encoder_seq_length, max_decoder_seq_length,
                                      num_encoder_tokens, num_decoder_tokens, input_token_index, target_token_index)

    print("TRAINING====================================")    
    model, encoder_model, decoder_model = define_model(num_encoder_tokens, num_decoder_tokens, input_params.num_units)

    # Run training
    # early_stop = EarlyStopping(monitor='val_loss', verbose=1)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['mse'])
    model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
              verbose=2, batch_size=input_params.batch_size, #callbacks=[early_stop],
              epochs=input_params.epochs,
              validation_split=0.25, shuffle=True)

    # Save all models
    # os.makedirs('models', exist_ok=True)
    # encoder_model, decoder_model = save_trained_models(model, encoder_model, decoder_model, num_units)
    print("TEST====================================")
    eval_test_set(encoder_model, decoder_model, input_texts, target_texts, encoder_input_data,
                      num_encoder_tokens, num_decoder_tokens,
                      target_token_index, max_decoder_seq_length)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--mode", type=str, dest='mode', default='infer',
                        # choices=['train', 'infer'],
                        # help='Either "train" for training or "infer" for inference')
    parser.add_argument("--batch", type=int, dest='batch_size', default=256,
                        help='Batch size')
    parser.add_argument("--epochs", type=int, dest='epochs', default=100,
                        help='Number of epochs')
    parser.add_argument("--units", type=int, dest='num_units', default=None,
                        help='Number of LSTM units')
    parser.add_argument("--enc_dim", type=int, dest='enc_dim', default=None,
                        help='Number of encoder tokens')
    parser.add_argument("--dec_dim", type=int, dest='dec_dim', default=None,
                        help='Number of decoder tokens')
    parser.add_argument("--datapath", type=Path, dest='datapath', default='data/fra.txt', #'data/fra_shuffled.txt'
                        help='Path to data [default: data/fra.txt]')
    parser.add_argument("--shuffle_data", action='store_true',
                        help="Shuffle input data and save")
    parser.add_argument("--numtrain", type=float, dest='numtrain', default=10000,
                        help='Number of samples for training') #val: 0.25 of train samp and test are train+val 
    args = parser.parse_args()

    # Only shuffle data for training
    # args.shuffle_data = args.shuffle_data if args.mode == 'train' else False
    args.datapath = os.path.join(os.path.realpath(os.getcwd()), str(args.datapath))
    args.num_lines = sum(1 for line in io.open(args.datapath, 'r', encoding='utf-8'))
    
    # df = pd.DataFrame(columns=[])
    if all(arg is None for arg in [args.num_units, args.enc_dim, args.dec_dim]):
        for num_units in [128, 256, 512]:
            for enc_dim in [32, 64, 71, 93, 128]:
                for dec_dim in [32, 64, 71, 93, 128]:
                    args.num_units, args.enc_dim, args.dec_dim = num_units, enc_dim, dec_dim
                    main(args)
    else:
        args.num_units = args.num_units if args.num_units is not None else 256
        args.enc_dim = args.enc_dim if args.enc_dim is not None else 71
        args.dec_dim = args.dec_dim if args.dec_dim is not None else 93
        main(args)
