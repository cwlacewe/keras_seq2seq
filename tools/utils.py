from __future__ import print_function
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
import numpy as np
import os
import argparse
import random
import io
from pathlib import Path


datalength = 145437

def vectorize_data(data):
    input_texts = []
    target_texts = []
    input_characters = set()
    target_characters = set()
    for line in data:
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
    input_token_index = dict(
        [(char, i) for i, char in enumerate(input_characters)])
    target_token_index = dict(
        [(char, i) for i, char in enumerate(target_characters)]) 
    
    # Reverse-lookup token index to decode sequences back to
    # something readable.
    # reverse_input_char_index = dict(
        # (i, char) for char, i in input_token_index.items())
    # reverse_target_char_index = dict(
        # (i, char) for char, i in target_token_index.items())       
    
    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)
    max_encoder_seq_length = max([len(txt) for txt in input_texts]) #Max sentence length
    max_decoder_seq_length = max([len(txt) for txt in target_texts]) #Max sentence length
        
    return input_texts, target_texts, input_characters, target_characters, \
        input_token_index, target_token_index, num_encoder_tokens, num_decoder_tokens, \
        max_encoder_seq_length, max_decoder_seq_length

def save_shuffle_data(data_path):
    input_texts = []
    target_texts = []
    input_characters = set() # unique chars 
    target_characters = set() # unique chars 
    out_data_path = data_path.split('.txt')[0] + '_shuffled.txt'
    with io.open(data_path, 'r', encoding='utf-8') as f:
        data_tmp = [(random.random(), line.splitlines()[0]) for line in f]
        data_tmp.sort()
        data = [line[1] for line in data_tmp]
        input_texts, target_texts, input_characters, target_characters, input_token_index, \
            target_token_index, num_encoder_tokens, num_decoder_tokens, \
            max_encoder_seq_length, max_decoder_seq_length = vectorize_data(data)
            
    with io.open(out_data_path, 'w', encoding='utf-8') as f:
        f.writelines('\n'.join(data))
        
    print('Data path:', data_path)
    print('Number of samples:', len(input_texts_all))
    print('Number of training samples:', len(input_texts_all))
    print('Number of validation samples:', len(input_texts_all))
    print('Number of testing samples:', len(input_texts_all))
    print('Number of unique input tokens:', num_encoder_tokens)
    print('Number of unique output tokens:', num_decoder_tokens)
    print('Max sequence length for inputs:', max_encoder_seq_length)
    print('Max sequence length for outputs:', max_decoder_seq_length)
    return data, input_texts, target_texts, \
        input_token_index, target_token_index, num_encoder_tokens, num_decoder_tokens, \
        max_encoder_seq_length, max_decoder_seq_length
    
def read_data(data_path):
    with io.open(data_path, 'r', encoding='utf-8') as f:
        data = [line.splitlines()[0] for line in f]
        input_texts, target_texts, input_characters, target_characters, input_token_index, \
        target_token_index, num_encoder_tokens, num_decoder_tokens, \
        max_encoder_seq_length, max_decoder_seq_length = vectorize_data(data)
    # print('Data path:', data_path)
    # print('Number of samples:', len(input_texts))
    # print('Number of unique input tokens:', num_encoder_tokens)
    # print('Number of unique output tokens:', num_decoder_tokens)
    # print('Max sequence length for inputs:', max_encoder_seq_length)
    # print('Max sequence length for outputs:', max_decoder_seq_length)
    return data, input_texts, target_texts, \
        input_token_index, target_token_index, num_encoder_tokens, num_decoder_tokens, \
        max_encoder_seq_length, max_decoder_seq_length
    
def prepare_data(data_path, train_percentage, shuffle=True):
    # Read data
    if shuffle:
        data, input_texts_all, target_texts_all, \
        input_token_index, target_token_index, num_encoder_tokens, num_decoder_tokens, \
        max_encoder_seq_length, max_decoder_seq_length = save_shuffle_data(data_path)
    else:
        data, input_texts_all, target_texts_all, \
        input_token_index, target_token_index, num_encoder_tokens, num_decoder_tokens, \
        max_encoder_seq_length, max_decoder_seq_length = read_data(data_path)

    num_trn_samples = 10000 #int(train_percentage * len(data))
        
    num_samples_per_set = int(len(data[num_trn_samples:]) / 2)
    train_input_texts, train_target_texts = input_texts_all[:num_trn_samples], target_texts_all[:num_trn_samples]
    val_input_texts, val_target_texts = input_texts_all[num_trn_samples : 2 * num_trn_samples], target_texts_all[num_trn_samples : 2 * num_trn_samples]
    test_input_texts, test_target_texts = input_texts_all[2 * num_trn_samples: 3 * num_trn_samples], target_texts_all[2 * num_trn_samples: 3 * num_trn_samples]
    # val_input_texts, val_target_texts = input_texts_all[num_trn_samples : num_trn_samples + num_samples_per_set], target_texts_all[num_trn_samples : num_trn_samples + num_samples_per_set]
    # test_input_texts, test_target_texts = input_texts_all[num_trn_samples + num_samples_per_set:], target_texts_all[num_trn_samples + num_samples_per_set:]
    
    input_characters = set()
    target_characters = set()
    # for line in data:
    for input_text, target_text in zip(train_input_texts, train_target_texts):
        # We use "tab" as the "start sequence" character
        # for the targets, and "\n" as "end sequence" character.
        # target_text = '\t' + target_text + '\n'
        # input_texts.append(input_text)
        # target_texts.append(target_text)
        for char in input_text:
            if char not in input_characters:
                input_characters.add(char)
        for char in target_text:
            if char not in target_characters:
                target_characters.add(char)
    input_characters = sorted(list(input_characters))
    target_characters = sorted(list(target_characters))
    input_token_index = dict(
        [(char, i) for i, char in enumerate(input_characters)])
    target_token_index = dict(
        [(char, i) for i, char in enumerate(target_characters)])     
    
    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)
    max_encoder_seq_length = max([len(txt) for txt in train_input_texts]) #Max sentence length
    max_decoder_seq_length = max([len(txt) for txt in train_target_texts]) #Max sentence length
    print('Data path:', data_path)
    print('Number of samples:', len(train_input_texts))
    print('Number of unique input tokens:', num_encoder_tokens)
    print('Number of unique output tokens:', num_decoder_tokens)
    print('Max sequence length for inputs:', max_encoder_seq_length)
    print('Max sequence length for outputs:', max_decoder_seq_length)
            
    print('Number of training samples:', len(train_input_texts))
    print('Number of validation samples:', len(val_input_texts))
    print('Number of testing samples:', len(test_input_texts))
    return max_encoder_seq_length, num_encoder_tokens, max_decoder_seq_length, num_decoder_tokens, \
        input_token_index, target_token_index, train_input_texts, train_target_texts, \
        val_input_texts, val_target_texts, test_input_texts, test_target_texts

   
def get_input_and_target_data(input_sentences, target_sentences, max_encoder_seq_length, max_decoder_seq_length,
                              num_encoder_tokens, num_decoder_tokens,
                              input_token_index, target_token_index):
    encoder_input_data = np.zeros(
        (len(input_sentences), max_encoder_seq_length, num_encoder_tokens),
        dtype='float32')
    decoder_input_data = np.zeros(
        (len(target_sentences), max_decoder_seq_length, num_decoder_tokens),
        dtype='float32')
    decoder_target_data = np.zeros(
        (len(target_sentences), max_decoder_seq_length, num_decoder_tokens),
        dtype='float32')
    for i, (input_text, target_text) in enumerate(zip(input_sentences, target_sentences)):
        for t, char in enumerate(input_text):
            char_idx = input_token_index[char]
            if char_idx < num_encoder_tokens:
                encoder_input_data[i, t, char_idx] = 1.
        for t, char in enumerate(target_text):
            char_idx = target_token_index[char]
            if char_idx < num_decoder_tokens:
                # decoder_target_data is ahead of decoder_input_data by one timestep
                decoder_input_data[i, t, char_idx] = 1.
                if t > 0:
                    # decoder_target_data will be ahead by one timestep
                    # and will not include the start character.
                    decoder_target_data[i, t - 1, char_idx] = 1.
    return encoder_input_data, decoder_input_data, decoder_target_data
    

def define_model(num_encoder_tokens, num_decoder_tokens, n_units):
    # define training encoder
    encoder_inputs = Input(shape=(None, num_encoder_tokens))
    encoder = LSTM(n_units, return_state=True, name='encoder_lstm')
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]
    encoder_model = Model(encoder_inputs, encoder_states)
    
    # define training decoder
    decoder_inputs = Input(shape=(None, num_decoder_tokens))
    decoder_lstm = LSTM(n_units, name='decoder_lstm', return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(num_decoder_tokens, name='decoder_dense', activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)  

    decoder_state_input_h = Input(shape=(n_units,))
    decoder_state_input_c = Input(shape=(n_units,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)
    return model, encoder_model, decoder_model
    
    
def save_trained_models(model, encoder_model, decoder_model, num_units):
    encoder_model.get_layer('encoder_lstm').set_weights(model.get_layer('encoder_lstm').get_weights())
    decoder_model.get_layer('decoder_lstm').set_weights(model.get_layer('decoder_lstm').get_weights())
    decoder_model.get_layer('decoder_dense').set_weights(model.get_layer('decoder_dense').get_weights())
    encoder_model.save('models/seq2seq_encoder_{}units.h5'.format(num_units))
    decoder_model.save('models/seq2seq_decoder_{}units.h5'.format(num_units))
    model.save('models/seq2seq_training_model_{}units.h5'.format(num_units))
    return encoder_model, decoder_model


def load_defined_models(n_units):
    # Save all models
    # model = load_model('models/seq2seq_training_model_{}units.h5'.format(n_units))
    encoder_model = load_model('models/seq2seq_encoder_{}units.h5'.format(n_units))
    decoder_model = load_model('models/seq2seq_decoder_{}units.h5'.format(n_units))
    
    return encoder_model, decoder_model


def predict_sequence(encoder_model, decoder_model, in_seq, n_output, target_token_ind,
                     max_decoder_seq_len):
    reverse_target_char_ind = dict(
        (i, char) for char, i in target_token_ind.items())   
    # encode
    state = encoder_model.predict(in_seq)
    # start of sequence input
    target_seq = np.array([0.0 for _ in range(n_output)]).reshape(1, 1, n_output)
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_ind['\t']] = 1.
    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        # predict next char
        yhat, h, c = decoder_model.predict([target_seq] + state)

        # Sample a token
        sampled_token_index = np.argmax(yhat[0, 0, :])
        sampled_char = reverse_target_char_ind[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '\n' or
                len(decoded_sentence) > max_decoder_seq_len):
            stop_condition = True

        # Update the target sequence (of length 1).
        # target_seq = np.array([0.0 for _ in range(n_output)]).reshape(1, 1, n_output)
        # target_seq[0, 0, sampled_token_index] = 1.
        target_seq = yhat

        # Update states
        state = [h, c]

    return decoded_sentence

                          
def eval_test_set(encoder_model, decoder_model, test_input_texts, test_target_texts, test_encoder_input_data,
                  num_encoder_tokens, num_decoder_tokens,
                  target_token_index, max_decoder_seq_length):
    in_texts = []
    actual_texts = []
    predicted_texts = []
    total_bleu = 0
    # test_input_texts, test_target_texts
    # test_encoder_input_data, test_decoder_input_data, test_decoder_target_data
    for seq_index in range(len(test_input_texts)):
        # Take one sequence (part of the training test)
        # for trying out decoding.
        input_seq = test_encoder_input_data[seq_index: seq_index + 1]
        if len(input_seq) <= num_encoder_tokens:
            try:
                decoded_sentence = predict_sequence(encoder_model, decoder_model, input_seq, num_decoder_tokens,
                                                    target_token_index,
                                                    max_decoder_seq_length)
                txt_actual = test_target_texts[seq_index].replace('\t','').replace('\n','').split()
                txt_pred = decoded_sentence.replace('\t','').replace('\n','').split()
                bleu = sentence_bleu([txt_actual],txt_pred)
                total_bleu += bleu
                actual_texts.append(txt_actual)
                # in_texts.append(test_input_texts[seq_index])
                predicted_texts.append(txt_pred)
                print('TEST SAMPLE {}===================================='.format(seq_index))
                print('Input sentence:', test_input_texts[seq_index])
                print('Decoded sentence:', decoded_sentence.replace('\t','').replace('\n',''))
                print('Target sentence:', test_target_texts[seq_index].replace('\t','').replace('\n',''))
                print('Bleu:', bleu)
            except:
                pass
    # Bleu Scores
    # print('Bleu-1: %f' % corpus_bleu(actual_texts, predicted_texts, weights=(1.0, 0, 0, 0)))
    # print('Bleu-2: %f' % corpus_bleu(actual_texts, predicted_texts, weights=(0.5, 0.5, 0, 0)))
    # print('Bleu-3: %f' % corpus_bleu(actual_texts, predicted_texts, weights=(0.3, 0.3, 0.3, 0)))
    # print('Bleu-4: %f' % corpus_bleu(actual_texts, predicted_texts, weights=(0.25, 0.25, 0.25, 0.25)))
    avg_bleu = (total_bleu / len(actual_texts))
    print('Avg. Bleu: %f' % avg_bleu)
    return avg_bleu
