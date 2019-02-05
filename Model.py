import tensorflow as tf

class model():            
    def enc_dec_model_inputs(self):
        inputs = tf.placeholder(tf.int32, [None, None], name='input')
        targets = tf.placeholder(tf.int32, [None, None], name='targets') 
        
        target_sequence_length = tf.placeholder(tf.int32, [None], name='target_sequence_length')
        source_sequence_length = tf.placeholder(tf.int32, [None], name='target_sequence_length')
        max_target_len = tf.reduce_max(target_sequence_length)    
        
        return inputs, targets, target_sequence_length, max_target_len, source_sequence_length
    
    def hyperparam_inputs(self):
        lr_rate = tf.placeholder(tf.float32, name='lr_rate')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        
        return lr_rate, keep_prob
    
    def process_decoder_input(self, target_data, target_vocab_to_int, batch_size):
        go_id = target_vocab_to_int['<GO>']
        
        after_slice = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
        after_concat = tf.concat( [tf.fill([batch_size, 1], go_id), after_slice], 1)
        
        return after_concat
    
    
    def encoding_layer(self, rnn_inputs, rnn_size, num_layers, keep_prob, 
                       source_vocab_size, 
                       encoding_embedding_size,
                       source_sequence_length,
                       emb_matrix):
        
        embed = tf.nn.embedding_lookup(emb_matrix, rnn_inputs)
        
        stacked_cells = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(rnn_size), keep_prob) for _ in range(num_layers)])
        
        outputs, state = tf.nn.bidirectional_dynamic_rnn(cell_fw=stacked_cells, 
                                                                 cell_bw=stacked_cells, 
                                                                 inputs=embed, 
                                                                 sequence_length=source_sequence_length, 
                                                                 dtype=tf.float32)
        
        concat_outputs = tf.concat(outputs, 2)
        return concat_outputs, state
    
    def decoding_layer_train(self, encoder_outputs, encoder_state, dec_cell, dec_embed_input, 
                             target_sequence_length, max_summary_length, 
                             output_layer, keep_prob, rnn_size, batch_size):
        """
        Create a training process in decoding layer 
        :return: BasicDecoderOutput containing training logits and sample_id
        """
        
        dec_cell = tf.contrib.rnn.DropoutWrapper(dec_cell, 
                                                 output_keep_prob=keep_prob)
        
        
        train_helper = tf.contrib.seq2seq.TrainingHelper(dec_embed_input, target_sequence_length)
        
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(rnn_size, encoder_outputs,
                                                                   memory_sequence_length=target_sequence_length)
        
        attention_cell = tf.contrib.seq2seq.AttentionWrapper(dec_cell, attention_mechanism,
                                                             attention_layer_size=rnn_size/2)
        
        decoder = tf.contrib.seq2seq.BasicDecoder(cell=attention_cell, helper=train_helper, 
                                                  initial_state=attention_cell.zero_state(dtype=tf.float32, batch_size=batch_size),
                                                  output_layer=output_layer) 
        outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, impute_finished=True, maximum_iterations=max_summary_length)
        
        return outputs
    
    
    
    def decoding_layer_infer(self, encoder_outputs, encoder_state, dec_cell,
                             dec_embeddings, start_of_sequence_id,
                             end_of_sequence_id, max_target_sequence_length,
                             vocab_size, output_layer, batch_size, keep_prob,
                             target_sequence_length, rnn_size):
        """
        Create a inference process in decoding layer 
        :return: BasicDecoderOutput containing inference logits and sample_id
        """
        dec_cell = tf.contrib.rnn.DropoutWrapper(dec_cell, 
                                                 output_keep_prob=keep_prob)
        
        infer_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(dec_embeddings, 
                                                          tf.fill([batch_size], start_of_sequence_id), 
                                                          end_of_sequence_id)
        
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(rnn_size, encoder_outputs,
                                                                   memory_sequence_length=target_sequence_length)
        
        attention_cell = tf.contrib.seq2seq.AttentionWrapper(dec_cell, attention_mechanism,
                                                             attention_layer_size=rnn_size/2)
        
        decoder = tf.contrib.seq2seq.BasicDecoder(cell=attention_cell, helper=infer_helper, 
                                                  initial_state=attention_cell.zero_state(dtype=tf.float32, batch_size=batch_size),
                                                  output_layer=output_layer)
       
        outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, impute_finished=True, maximum_iterations=max_target_sequence_length)
        
        return outputs
    
    def decoding_layer(self, encoder_outputs, dec_input, encoder_state,
                       target_sequence_length, max_target_sequence_length,
                       rnn_size,
                       num_layers, target_vocab_to_int, target_vocab_size,
                       batch_size, keep_prob, decoding_embedding_size,
                       emb_matrix):
        """
        Create decoding layer
        :return: Tuple of (Training BasicDecoderOutput, Inference BasicDecoderOutput)
        """
        target_vocab_size = len(target_vocab_to_int) + 1
        dec_embed_input = tf.nn.embedding_lookup(emb_matrix, dec_input)
        
        cells = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(rnn_size) for _ in range(num_layers)])
        
        with tf.variable_scope("decode"):
            output_layer = tf.layers.Dense(target_vocab_size)
            train_output = self.decoding_layer_train(encoder_outputs,
                                                encoder_state, 
                                                cells, 
                                                dec_embed_input, 
                                                target_sequence_length, 
                                                max_target_sequence_length, 
                                                output_layer, 
                                                keep_prob,
                                                rnn_size,
                                                batch_size)
    
        with tf.variable_scope("decode", reuse=True):
            infer_output = self.decoding_layer_infer(encoder_outputs,
                                                encoder_state, 
                                                cells, 
                                                emb_matrix, 
                                                target_vocab_to_int['<GO>'], 
                                                target_vocab_to_int['<EOS>'], 
                                                max_target_sequence_length, 
                                                target_vocab_size, 
                                                output_layer,
                                                batch_size,
                                                keep_prob,
                                                target_sequence_length,
                                                rnn_size)
    
        return (train_output, infer_output)
    
    def seq2seq(self,  input_data, target_data, keep_prob, batch_size,
                      target_sequence_length,
                      max_target_sentence_length,
                      source_vocab_size, target_vocab_size,
                      enc_embedding_size, dec_embedding_size,
                      rnn_size, num_layers, target_vocab_to_int,
                      source_sequence_length,
                      emb_matrix):
        """
        Build the Sequence-to-Sequence model
        :return: Tuple of (Training BasicDecoderOutput, Inference BasicDecoderOutput)
        """
        enc_outputs, enc_states = self.encoding_layer(input_data, 
                                                 rnn_size, 
                                                 num_layers, 
                                                 keep_prob, 
                                                 source_vocab_size, 
                                                 enc_embedding_size,
                                                 source_sequence_length,
                                                 emb_matrix)
        
        dec_input = self.process_decoder_input(target_data, 
                                          target_vocab_to_int, 
                                          batch_size)
        
        train_output, infer_output = self.decoding_layer(enc_outputs,
                                                    dec_input,
                                                   enc_states, 
                                                   target_sequence_length, 
                                                   max_target_sentence_length,
                                                   rnn_size,
                                                  num_layers,
                                                  target_vocab_to_int,
                                                  target_vocab_size,
                                                  batch_size,
                                                  keep_prob,
                                                  dec_embedding_size,
                                                  emb_matrix)
        
        return train_output, infer_output