import pandas as pd
import numpy as np
import tensorflow as tf
import os
from os import listdir
from os.path import isfile, join
from collections import namedtuple
from tensorflow.python.layers.core import Dense
from tensorflow.python.ops.rnn_cell_impl import _zero_state_tensors
import time
import re
from sklearn.model_selection import train_test_split
import json
import difflib
from parameters import *


def load_book(path):
    """Load a book from its file"""
    input_file = os.path.join(path)
    with open(input_file) as f:
        book = f.read()
    return book

def clean_text(text):
    #Remove unwanted characters and extra spaces from the text
    text = re.sub(r'\n', ' ', text) 
    text = re.sub(r'[{}@_*>()\\#%+=\[\]]','', text)
    text = re.sub('a0','', text)
    text = re.sub('\'92t','\'t', text)
    text = re.sub('\'92s','\'s', text)
    text = re.sub('\'92m','\'m', text)
    text = re.sub('\'92ll','\'ll', text)
    text = re.sub('\'91','', text)
    text = re.sub('\'92','', text)
    text = re.sub('\'93','', text)
    text = re.sub('\'94','', text)
    text = re.sub('\.','. ', text)
    text = re.sub('\!','', text)
    text = re.sub('\?','', text)
    text = re.sub(' +',' ', text)
    text = re.sub(',','', text)
    text = re.sub('-','', text)
    text = re.sub('; ','', text)
    text = re.sub(':','', text)
    text = re.sub('"','', text)
    text = re.sub("'97",'\'', text)
    return text

def noise_maker(sentence, threshold):
    '''Relocate, remove, or add characters to create spelling mistakes'''
    
    noisy_sentence = []
    i = 0
    while i < len(sentence):
        random = np.random.uniform(0,1,1)
        # Most characters will be correct since the threshold value is high
        if random < threshold:
            noisy_sentence.append(sentence[i])
        else:
            new_random = np.random.uniform(0,1,1)
            # ~33% chance characters will swap locations
            if new_random > 0.67:
                if i == (len(sentence) - 1):
                    # If last character in sentence, it will not be typed
                    continue
                else:
                    # if any other character, swap order with following character
                    noisy_sentence.append(sentence[i+1])
                    noisy_sentence.append(sentence[i])
                    i += 1
            # ~33% chance an extra lower case letter will be added to the sentence
            elif new_random < 0.33:
                random_letter = np.random.choice(letters, 1)[0]
                noisy_sentence.append(vocab_to_int[random_letter])
                noisy_sentence.append(sentence[i])
            # ~33% chance a character will not be typed
            else:
                pass     
        i += 1
    return noisy_sentence

# Check to ensure noise_maker is making mistakes correctly.

#for sentence in training_sorted[:5]:
    #print(sentence)
    #print(noise_maker(sentence, threshold))
    #print()

def model_inputs():
  
    #tf.device("/device:GPU:0")
    '''Create palceholders for inputs to the model'''
    
    with tf.name_scope('inputs'):
        inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
    with tf.name_scope('targets'):
        targets = tf.placeholder(tf.int32, [None, None], name='targets')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    inputs_length = tf.placeholder(tf.int32, (None,), name='inputs_length')
    targets_length = tf.placeholder(tf.int32, (None,), name='targets_length')
    #targets_length = tf.Variable([2, 3, 5, 7, 11], tf.int32)
    max_target_length = tf.reduce_max(targets_length, name='max_target_len')
    #max_target_length = tf.to_int32(max_target_len, name='ToInt32')
    #max_target_length =

    
    #print("max_length:", tf.Print(max_target_length,[max_target_length]))
   

    return inputs, targets, keep_prob, inputs_length, targets_length ,max_target_length  #tf.reduce_max(targets_length,name='max_target_len')

def process_encoding_input(targets, vocab_to_int, batch_size):
    '''Remove the last word id from each batch and concat the <GO> to the begining of each batch'''
    
    with tf.name_scope("process_encoding"):
        ending = tf.strided_slice(targets, [0, 0], [batch_size, -1], [1, 1])
        dec_input = tf.concat([tf.fill([batch_size, 1], vocab_to_int['<GO>']), ending], 1)

    return dec_input

def encoding_layer(rnn_size, sequence_length, num_layers, rnn_inputs, keep_prob, direction):
    '''Create the encoding layer'''
    
    if direction == 1:
        with tf.name_scope("RNN_Encoder_Cell_1D"):
            for layer in range(num_layers):
                with tf.variable_scope('encoder_{}'.format(layer)):
                    lstm = tf.contrib.rnn.LSTMCell(rnn_size)

                    drop = tf.contrib.rnn.DropoutWrapper(lstm, 
                                                         input_keep_prob = keep_prob)

                    enc_output, enc_state = tf.nn.dynamic_rnn(drop, 
                                                              rnn_inputs,
                                                              sequence_length,
                                                              dtype=tf.float32)

            return enc_output, enc_state
        
        
    if direction == 2:
        with tf.name_scope("RNN_Encoder_Cell_2D"):
            for layer in range(num_layers):
                with tf.variable_scope('encoder_{}'.format(layer)):
                    cell_fw = tf.contrib.rnn.LSTMCell(rnn_size)
                    cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, 
                                                            input_keep_prob = keep_prob)

                    cell_bw = tf.contrib.rnn.LSTMCell(rnn_size)
                    cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, 
                                                            input_keep_prob = keep_prob)

                    enc_output, enc_state = tf.nn.bidirectional_dynamic_rnn(cell_fw, 
                                                                            cell_bw, 
                                                                            rnn_inputs,
                                                                            sequence_length,
                                                                            dtype=tf.float32)
            # Join outputs since we are using a bidirectional RNN
            enc_output = tf.concat(enc_output,2)
            # Use only the forward state because the model can't use both states at once
            return enc_output, enc_state[0]

def training_decoding_layer(dec_embed_input, targets_length, dec_cell, initial_state, output_layer, 
                            vocab_size):
    '''Create the training logits'''
    
    with tf.name_scope("Training_Decoder"):
        training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=dec_embed_input,
                                                            sequence_length=targets_length,
                                                            time_major=False)

        training_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                           training_helper,
                                                           initial_state,
                                                           output_layer) 
        
        #initial_state = dec_cell.zero_state(dtype=tf.float32, batch_size=batch_size)

        training_logits, one,two = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                                     output_time_major=False,
                                                                     impute_finished=True,
                                                                     maximum_iterations=tf.reduce_max(targets_length))
        return training_logits

def inference_decoding_layer(embeddings, start_token, end_token, dec_cell, initial_state, output_layer,max_target_length,
                             batch_size,targets_length):
    '''Create the inference logits'''
    
    with tf.name_scope("Inference_Decoder"):
        start_tokens = tf.tile(tf.constant([start_token], dtype=tf.int32), [batch_size], name='start_tokens')

        inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embeddings,
                                                                    start_tokens,
                                                                    end_token)

        inference_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                            inference_helper,
                                                            initial_state,
                                                            output_layer)

        inference_logits, one_in,two_in = tf.contrib.seq2seq.dynamic_decode(inference_decoder,
                                                                            output_time_major=False,
                                                                            impute_finished=True,
                                                                            maximum_iterations=tf.reduce_max(targets_length))

        return inference_logits

def decoding_layer(dec_embed_input, embeddings, enc_output, enc_state, vocab_size, inputs_length, targets_length, max_target_length,
                   rnn_size, vocab_to_int, keep_prob, batch_size, num_layers,direction):
    '''Create the decoding cell and attention for the training and inference decoding layers'''
    
    with tf.name_scope("RNN_Decoder_Cell"):
        for layer in range(num_layers):
            with tf.variable_scope('decoder_{}'.format(layer)):
                lstm = tf.contrib.rnn.LSTMCell(rnn_size)
                dec_cell = tf.contrib.rnn.DropoutWrapper(lstm, 
                                                         input_keep_prob = keep_prob)
    
    output_layer = Dense(vocab_size,
                         kernel_initializer = tf.truncated_normal_initializer(mean = 0.0, stddev=0.1))
    
    attn_mech = tf.contrib.seq2seq.BahdanauAttention(rnn_size,
                                                  enc_output,
                                                  inputs_length,
                                                  normalize=False,
                                                  name='BahdanauAttention')
    
    with tf.name_scope("Attention_Wrapper"):
        dec_cell = tf.contrib.seq2seq.AttentionWrapper(dec_cell,
                                                              attn_mech,
                                                              rnn_size)
    
    #initial_state = tf.contrib.seq2seq.AttentionWrapperState(enc_state,_zero_state_tensors(rnn_size, batch_size, tf.float32))
    
    #initial_state = dec_cell.zero_state(dtype=tf.float32, batch_size=batch_size)
    
#     attn_zero = dec_cell.zero_state(batch_size , tf.float32 )

#     attn_zero = attn_zero.clone(cell_state = enc_state)
                                                                                        
#     initial_state = tf.contrib.seq2seq.AttentionWrapperState(cell_state = enc_state, 
#                                                               attention = attn_zero,
#                                                               time = 0,
#                                                               alignments=attn_mech,
#                                                               alignment_history=None,
#                                                               attention_state = dec_cell)

#     #initial_state = enc_state
  
    initial_state = dec_cell.zero_state(dtype=tf.float32, batch_size=batch_size).clone(cell_state=enc_state)

    with tf.variable_scope("decode"):
        training_logits = training_decoding_layer(dec_embed_input, 
                                                  targets_length, 
                                                  dec_cell, 
                                                  initial_state,
                                                  output_layer,
                                                  vocab_size)
    with tf.variable_scope("decode", reuse=True):
        inference_logits = inference_decoding_layer(embeddings,  
                                                    vocab_to_int['<GO>'], 
                                                    vocab_to_int['<EOS>'],
                                                    dec_cell, 
                                                    initial_state, 
                                                    output_layer,
                                                    max_target_length,
                                                    batch_size,
                                                   targets_length)

    return training_logits, inference_logits

def seq2seq_model(inputs, targets, keep_prob, inputs_length, targets_length,max_target_length, 
                  vocab_size, rnn_size, num_layers, vocab_to_int, batch_size, embedding_size,direction):
    '''Use the previous functions to create the training and inference logits'''
    
    enc_embeddings = tf.Variable(tf.random_uniform(shape=[vocab_size, embedding_size], minval = -1, maxval = 1, seed = 0.5))
    enc_embed_input = tf.nn.embedding_lookup(enc_embeddings, inputs)
    enc_output, enc_state = encoding_layer(rnn_size, inputs_length, num_layers, 
                                           enc_embed_input, keep_prob,direction)
    
    dec_embeddings = tf.Variable(tf.random_uniform(shape=[vocab_size, embedding_size],minval=-1,maxval= 1,seed = 0.5))
    dec_input = process_encoding_input(targets, vocab_to_int, batch_size)
    dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)
    
    training_logits, inference_logits  = decoding_layer(dec_embed_input, 
                                                        dec_embeddings,
                                                        enc_output,
                                                        enc_state, 
                                                        vocab_size, 
                                                        inputs_length, 
                                                        targets_length, 
                                                        max_target_length,
                                                        rnn_size, 
                                                        vocab_to_int, 
                                                        keep_prob, 
                                                        batch_size,
                                                        num_layers,
                                                        direction)
    
    return training_logits, inference_logits

def pad_sentence_batch(sentence_batch):
    """Pad sentences with <PAD> so that each sentence of a batch has the same length"""
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [vocab_to_int['<PAD>']] * (max_sentence - len(sentence)) for sentence in sentence_batch]

def get_batches(sentences, batch_size, threshold):
    """Batch sentences, noisy sentences, and the lengths of their sentences together.
       With each epoch, sentences will receive new mistakes"""
    
    for batch_i in range(0, len(sentences)//batch_size):
        start_i = batch_i * batch_size
        sentences_batch = sentences[start_i:start_i + batch_size]
        
        sentences_batch_noisy = []
        for sentence in sentences_batch:
            sentences_batch_noisy.append(noise_maker(sentence, threshold))
            
        sentences_batch_eos = []
        for sentence in sentences_batch:
            sentence.append(vocab_to_int['<EOS>'])
            sentences_batch_eos.append(sentence)
            
        pad_sentences_batch = np.array(pad_sentence_batch(sentences_batch_eos))
        pad_sentences_noisy_batch = np.array(pad_sentence_batch(sentences_batch_noisy))
        
        # Need the lengths for the _lengths parameters
        pad_sentences_lengths = []
        for sentence in pad_sentences_batch:
            pad_sentences_lengths.append(len(sentence))
        
        pad_sentences_noisy_lengths = []
        for sentence in pad_sentences_noisy_batch:
            pad_sentences_noisy_lengths.append(len(sentence))
        
        yield pad_sentences_noisy_batch, pad_sentences_batch, pad_sentences_noisy_lengths, pad_sentences_lengths

def build_graph(keep_prob, rnn_size, num_layers, batch_size, learning_rate, embedding_size,direction):

    tf.reset_default_graph()
    
    # Load the model inputs    
    inputs, targets, keep_prob, inputs_length, targets_length, max_target_length = model_inputs()

    # Create the training and inference logits
    training_logits, inference_logits = seq2seq_model(tf.reverse(inputs, [-1]),
                                                      targets, 
                                                      keep_prob,   
                                                      inputs_length,
                                                      targets_length,
                                                      max_target_length,
                                                      len(vocab_to_int)+1,
                                                      rnn_size, 
                                                      num_layers, 
                                                      vocab_to_int,
                                                      batch_size,
                                                      embedding_size,
                                                      direction)

    # Create tensors for the training logits and inference logits
    training_logits = tf.identity(training_logits.rnn_output, 'logits')

    with tf.name_scope('predictions'):
        predictions = tf.identity(inference_logits.sample_id, name='predictions')
        tf.summary.histogram('predictions', predictions)

    # Create the weights for sequence_loss
    masks = tf.sequence_mask(targets_length, dtype=tf.float32, name='masks')
    
    with tf.name_scope("cost"):
        # Loss function
        cost = tf.contrib.seq2seq.sequence_loss(training_logits, 
                                                targets, 
                                                masks)
        tf.summary.scalar('cost', cost)

    with tf.name_scope("optimze"):
        optimizer = tf.train.AdamOptimizer(learning_rate)

        # Gradient Clipping
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)

    # Merge all of the summaries
    merged = tf.summary.merge_all()    

    # Export the nodes 
    export_nodes = ['inputs', 'targets', 'keep_prob', 'cost', 'inputs_length', 'targets_length',
                    'predictions', 'merged', 'train_op','optimizer']
    Graph = namedtuple('Graph', export_nodes)
    local_dict = locals()
    graph = Graph(*[local_dict[each] for each in export_nodes])
    
    saver = tf.train.Saver()
    return graph, saver


# Train the model with the desired tuning parameters
'''for keep_probability in [0.75]:
    for num_layers in [3]:
        for threshold in [0.75]:
            log_string = 'kp={},nl={},th={}'.format(keep_probability,
                                                    num_layers,
                                                    threshold) 
            model, saver = build_graph(keep_probability, rnn_size, num_layers, batch_size, 
                                learning_rate, 
                                embedding_size,
                                direction)
            #train(model, epochs, log_string, saver)'''

def text_to_ints(text):
    '''Prepare the text for the model'''
    
    text = clean_text(text)
    return [vocab_to_int[word] for word in text]

path = './books/'
book_files = [f for f in listdir(path) if isfile(join(path, f))]
book_files = book_files[1:]

books = [] # books data ka array
for book in book_files:
    books.append(load_book(path+book))

# Clean the text of the books

clean_books = []

for book in books:
    book.lower()
    clean_books.append(clean_text(book))

# Create a dictionary to convert the vocabulary (characters) to integers
vocab_to_int = {}
'''count = 0
for book in clean_books:
    for character in book:
        if character not in vocab_to_int:
            vocab_to_int[character] = count
            count += 1'''
with open("./clean_data/vocab_to_int.json", 'r') as f:
        vocab_to_int = json.load(f)
count = len(vocab_to_int)
# Add special tokens to vocab_to_int
'''codes = ['<PAD>','<EOS>','<GO>']
for code in codes:
    vocab_to_int[code] = count
    count += 1'''

# Create another dictionary to convert integers to their respective characters
int_to_vocab = {}
for character, value in vocab_to_int.items():
    int_to_vocab[value] = character
#print(int_to_vocab)
# Split the text from the books into sentences.
sentences = []
'''for book in clean_books:
    for sentence in book.split('. '):
        sentences.append(sentence.lower())'''

text_file = open("./clean_data/sentences.txt",'r')
sentences = text_file.read().split(". ")
        
words_list = {}
for i in range(0,len(sentences)):
    temp_list = sentences[i].split(" ")
    for j in range(0,len(temp_list)):
        if temp_list[j] in words_list:
            val = words_list[temp_list[j]]
            val = val+1
            words_list[temp_list[j]] = val
        else:
            words_list[temp_list[j]] = 1

#for key,value in words_list.items():
#   print(key,value,"\n")

# Convert sentences to integers
int_sentences = []

for sentence in sentences:
    int_sentence = []
    for character in sentence:
        if character != "\n":
            int_sentence.append(vocab_to_int[character])
    int_sentences.append(int_sentence)

#print(int_sentences[0])

# Find the length of each sentence
lengths = []
for sentence in int_sentences:
    lengths.append(len(sentence))
lengths = pd.DataFrame(lengths, columns=["counts"])

lengths.describe()

max_length = 92
min_length = 10

good_sentences = []

for sentence in int_sentences:
    if len(sentence) <= max_length and len(sentence) >= min_length:
        good_sentences.append(sentence)

print("We will use {} to train and test our model.".format(len(good_sentences)))

# Split the data into training and testing sentences
training, testing = train_test_split(good_sentences, test_size = 0.15, random_state = 2)

print("Number of training sentences:", len(training))
print("Number of testing sentences:", len(testing))

# Sort the sentences by length to reduce padding, which will allow the model to train faster
training_sorted = []
testing_sorted = []

for i in range(min_length, max_length+1):
    for sentence in training:
        if len(sentence) == i:
            training_sorted.append(sentence)
    for sentence in testing:
        if len(sentence) == i:
            testing_sorted.append(sentence)

#used to modify sentences and create noise
letters = ['a','b','c','d','e','f','g','h','i','j','k','l','m',
           'n','o','p','q','r','s','t','u','v','w','x','y','z',]

