import tensorflow as tf
import tensorflow_addons as tfa
import pickle
def model_inputs():
    inputs = tf.compat.v1.placeholder(tf.int32,[None,None],name = 'input')
    targets = tf.compat.v1.placeholder(tf.int32, [None, None], name='target')
    lr = tf.compat.v1.placeholder(tf.float32,  name='learning_rate')
    keep_prob = tf.compat.v1.placeholder(tf.float32, name='keep_prob')
    return inputs,targets,lr,keep_prob


def preprocess_targets(targets,word2int, batch_size):
    left_side = tf.fill([batch_size,1], word2int['<SOS>'])

    right_side = tf.strided_slice(targets, [0,0], [batch_size, -1], [1,1])
    preprocessed_targets = tf.concat([left_side,right_side],1)
    return preprocessed_targets

#Encoding RNN layer
def encoder_rnn(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
    lstm = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(rnn_size)
    lstm_dropout = tfa.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
    encoder_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell([lstm_dropout]*num_layers)
    encoder_output, encoder_state = tf.compat.v1.nn.bidirectional_dynamic_rnn(cell_fw = encoder_cell,
                                                       cell_bw = encoder_cell,
                                                       sequence_length = sequence_length,
                                                       inputs = rnn_inputs,
                                                       dtyoe = tf.float32)

    return encoder_state

def decode_training_set(encoder_state, decoder_cell, decoder_embedded_input, sequence_length, decoding_scope, output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tfa.seq2seq.prepare_attention(attention_states, attention_option = 'bahdanau', num_units = decoder_cell.output_size)
    training_decoder_function = tfa.seq2seq.attention_decoder_fn_train(encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              name = 'attn_dec_train')
    decoder_output, decoder_final_state, decoder_final_context_state = tfa.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                              training_decoder_function,
                                                                                                              decoder_embedded_input,
                                                                                                              sequence_length,
                                                                                                              scope = decoding_scope)

    decoder_output_dropout = tf.nn.dropout(decoder_output,1 - (keep_prob))
    return output_function(decoder_output_dropout)


#Decoding the test/validation set
def decode_test_set(encoder_state, decoder_cell, decoder_embedded_matrix, sos_id, eos_id, maximum_length, num_words, sequence_length, decoding_scope, output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tfa.seq2seq.prepare_attention(attention_states, attention_option = 'bahdanau', num_units = decoder_cell.output_size)
    test_decoder_function = tfa.seq2seq.attention_decoder_fn_inference(output_function,
                                                                              encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              decoder_embedded_matrix,
                                                                              sos_id,
                                                                              eos_id,
                                                                              maximum_length,
                                                                              num_words,
                                                                              name = 'attn_dec_inf')
    test_predictions, decoder_final_state, decoder_final_context_state = tfa.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                                test_decoder_function,
                                                                                                                scope = decoding_scope)


    return test_predictions

def decoder_rnn(decoder_embedded_input, decoder_embeddings_matrix, encoder_state, num_words, sequence_length, rnn_size, num_layers, word2int, keep_prob, batch_size):
    with tf.compat.v1.variable_scope("decoding") as decoding_scope:
        lstm = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(rnn_size)
        lstm_dropout = tfa.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
        decoder_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell([lstm_dropout]*num_layers)
        weights = tf.compat.v1.truncated_normal_initializer(stddev = 0.1)
        biases = tf.compat.v1.zeros_initializer()
        output_function = lambda x: tfa.layers.fully_connected(x,
                                                                      num_words,
                                                                      None,
                                                                      scope = decoding_scope,
                                                                      weights_initializer = weights,
                                                                      biases_initializer = biases)
        training_predictions = decode_training_set(encoder_state,
                                                   decoder_cell,
                                                   decoder_embedded_input,
                                                   sequence_length,
                                                   decoding_scope,
                                                   output_function,
                                                   keep_prob,
                                                   batch_size)
        decoding_scope.reuse_variables()
        test_predictions = decode_test_set(encoder_state,
                                           decoder_cell,
                                           decoder_embeddings_matrix,
                                           word2int['<SOS>'],
                                           word2int['<EOS>'],
                                           sequence_length-1,
                                           num_words,
                                           decoding_scope,
                                           output_function,
                                           keep_prob,
                                           batch_size
                                           )

    return training_predictions, test_predictions

#Building seq2seq model

def seq2seq_model(inputs, targets, keep_prob, batch_size, sequence_length, answers_num_words, questions_num_words, encoder_embedding_size, decoder_embedding_size, rnn_size, num_layers, questionswords2int):
    encoder_embedded_input = tf.compat.v1.layers.embed_sequence(inputs,
                                                              answers_num_words+1,
                                                              encoder_embedding_size,
                                                              initializer = tf.compat.v1.random_uniform_initializer(0,1))
    encoder_state = encoder_rnn(encoder_embedded_input, rnn_size, num_layers, keep_prob, sequence_length)
    preprocessed_targets = preprocess_targets(targets, questionswords2int, batch_size)
    decoder_embeddings_matrix = tf.Variable(tf.random.uniform([questions_num_words+1, decoder_embedding_size], 0, 1))
    decoder_embedded_inputs = tf.nn.embedding_lookup(params=decoder_embeddings_matrix, ids=preprocessed_targets)
    training_predictions, test_predictions = decoder_rnn(decoder_embedded_inputs,
                                                         decoder_embeddings_matrix,
                                                         encoder_state,
                                                         questions_num_words,
                                                         sequence_length,
                                                         rnn_size,
                                                         num_layers,
                                                         questionswords2int,
                                                         keep_prob,
                                                         batch_size)

    return training_predictions, test_predictions

questions = pickle.load( open( "Pract2/ques.pkl", "rb" ) )
answers = pickle.load( open( "Pract2/ans.pkl", "rb" ) )
accuracy_score_lstm = pickle.load( open( "accuracy.pkl", "rb" ) )
confusion_matrix_lstm = pickle.load(open("confusion_matrix.pkl","rb"))
s1 = 'Accuracy_score\n'
s2 = 'Confusion_matrix\n'
'''
#Training the model
epochs = 100
batch_size = 64
rnn_size = 512
num_layers = 3
encoding_embedding_size = 512
decoding_embedding_size = 512
learning_rate = 0.01
learning_rate_decay = 0.9
min_learning_rate = 0.0001
keep_probability = 0.5

#defining tf session
tf.compat.v1.reset_default_graph()
session = tf.compat.v1.InteractiveSession()

#Loading model inputs
inputs, targets, lr, keep_prob = model_inputs()

#setting sequence length
sequence_length = tf.compat.v1.placeholder_with_default(25, None, name = "sequence_length")

#getting shape of input tensors
input_shape = tf.shape(input=inputs)

#getting the training and test predictions
training_predictions, test_predictions = seq2seq_model(tf.reverse(inputs,[-1]),
                                                       targets,
                                                       keep_prob,
                                                       batch_size,
                                                       sequence_length,
                                                       len(answerswords2int),
                                                       len(questionswords2int),
                                                       encoding_embedding_size,
                                                       decoding_embedding_size,
                                                       rnn_size,
                                                       num_layers,
                                                       questionswords2int)
'''


print(questions)
print(answers)
print(s1,accuracy_score_lstm)
print(s2,confusion_matrix_lstm)