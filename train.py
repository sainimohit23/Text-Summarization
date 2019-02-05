import pickle
import numpy as np
from Model import model
import tensorflow as tf
import datetime
from tqdm import tqdm


BATCH_SIZE = 64
RNN_SIZE = 128
NUM_LAYERS = 2
EMBEDDING_SIZE = 50
LEARNING_RATE = 0.001
LEARNING_RATE_DECAY = 0.9
MIN_LR = 0.0001
KEEP_PROB = 0.5
EPOCHS=10
DISPLAY_STEP=30
save_path = 'checkpoints/dev'


def load_preprocess(path, mode):
    with open(path, mode=mode) as in_file:
        return pickle.load(in_file)

article_list, title_list = load_preprocess('smallData.p', 'rb')
words_to_index, index_to_words, word_to_vec_map = load_preprocess('processedGlove.p','rb')


sourceMax=-100
for i in range(len(article_list)):
    if len(article_list[i]) > sourceMax:
        sourceMax = len(article_list[i]) 


emb_matrix = np.zeros((len(words_to_index)+1, word_to_vec_map['go'].shape[0]), dtype=np.float32)
for i in range(1, len(words_to_index)):
    emb_matrix[i] = word_to_vec_map[str(index_to_words[i])].astype(np.float32)

max_target_sentence_length = max([len(sentence) for sentence in title_list])

train_graph = tf.Graph()
with train_graph.as_default():
    myModel = model()
    inputs, targets, target_sequence_length, max_target_len, source_sequence_length = myModel.enc_dec_model_inputs()
    lr, keep_prob = myModel.hyperparam_inputs()
    
    train_logits, infer_logits = myModel.seq2seq(inputs,
                                                     targets,
                                                     KEEP_PROB,
                                                     BATCH_SIZE,
                                                     target_sequence_length,
                                                     max_target_len,
                                                     len(words_to_index),
                                                     len(words_to_index),
                                                     EMBEDDING_SIZE,
                                                     EMBEDDING_SIZE,
                                                     RNN_SIZE,
                                                     NUM_LAYERS,
                                                     words_to_index,
                                                     source_sequence_length,
                                                     emb_matrix)
    
    training_logits = tf.identity(train_logits.rnn_output, name='logits')
    inference_logits = tf.identity(infer_logits.sample_id, name='predictions')

    # https://www.tensorflow.org/api_docs/python/tf/sequence_mask
    # - Returns a mask tensor representing the first N positions of each cell.
    masks = tf.sequence_mask(target_sequence_length, max_target_len, dtype=tf.float32, name='masks')
    
    with tf.name_scope("optimization"):
        # Loss function - weighted softmax cross entropy
        cost = tf.contrib.seq2seq.sequence_loss(
            training_logits,
            targets,
            masks)


        optimizer = tf.train.AdamOptimizer(lr)

        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)
        
        tf.summary.scalar('loss', cost)
        merged = tf.summary.merge_all()
        logdir = 'tensorboard/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/" 
    

def pad_sentence_batch(sentance_batch, pad_int):
    max_len = max([len(sentance) for sentance in sentance_batch])
    return [sentance + [pad_int] * (max_len -len(sentance)) for sentance in sentance_batch], max_len


def get_batches(sources, targets, batch_size, pad_int):
    
    for batch_i in range(0, len(sources)//batch_size):
        start_i = batch_i * batch_size
        
        sources_batch = sources[start_i:start_i + batch_size]
        targets_batch = targets[start_i:start_i + batch_size]

        # Pad
        pad_sources_batch, max_source_len = np.array(pad_sentence_batch(sources_batch, pad_int))
        pad_targets_batch, max_target_len = np.array(pad_sentence_batch(targets_batch, pad_int))

        # Need the lengths for the _lengths parameters
        pad_targets_lengths = [max_target_len] * batch_size
        pad_source_lengths = [max_source_len] * batch_size 
        yield pad_sources_batch, pad_targets_batch, pad_source_lengths, pad_targets_lengths
        

def get_accuracy(target, logits):
    max_seq = max(len(target[1]), logits.shape[1])
    if max_seq - len(target[1]):
        target = np.pad(
            target,
            [(0,0),(0,max_seq - len(target[1]))],
            'constant')
    if max_seq - logits.shape[1]:
        logits = np.pad(
            logits,
            [(0,0),(0,max_seq - logits.shape[1])],
            'constant')

    return np.mean(np.equal(target, logits))


def sentence_to_seq(sentence, vocab_to_int):
    results = []
    for word in sentence.split(" "):
        if word in vocab_to_int:
            results.append(vocab_to_int[word])
        else:
            results.append(vocab_to_int['<UNK>'])
            
    return results


train_source = article_list[BATCH_SIZE:]
train_target = title_list[BATCH_SIZE:]
valid_source = article_list[:BATCH_SIZE]
valid_target = title_list[:BATCH_SIZE]
(valid_sources_batch, valid_targets_batch, valid_sources_lengths, valid_targets_lengths) = next(get_batches(valid_source,
                                                                                                         valid_target,
                                                                                                         BATCH_SIZE,
                                                                                                         words_to_index['<PAD>']))

sentances = ["fivetime world champion michelle kwan withdrew from the # us figure skating championships on wednesday  but will petition us skating officials for the chance to compete at the # turin olympics #",
             "jose mourinho renewed his partnership with portuguese international maniche on wednesday when he completed the loan signing of the #yearold midfielder from dynamo moscow #",
             "hollywood is planning a new sequel to adventure flick  ocean s eleven   with star george clooney set to reprise his role as a charismatic thief in  ocean s thirteen   the entertainment press said wednesday #",
             "teenage hollywood starlet lindsay lohan  who was rushed to hospital this week after suffering an asthma attack  has admitted fighting a fierce battle with the eating disorder bulimia #",
             "the us special envoy to multilateral talks aimed at ending north korea s nuclear weapons drive has quit  the state department said wednesday amid reported divisions within the administration over the nuclear issue #"]


with tf.Session(graph=train_graph) as sess:
    writer =  tf.summary.FileWriter(logdir, train_graph)
    saver = tf.train.Saver()
    try:
        saver.restore(sess, tf.train.latest_checkpoint('checkpoints/'))
        print("Saved model found")
    except ValueError:
        print("No saved model found, initializing new variables")
        sess.run(tf.global_variables_initializer())
        
    for epoch in tqdm(range(0, EPOCHS)):
        for batch_i, (source_batch, target_batch, sources_lengths, targets_lengths) in tqdm(enumerate(get_batches(train_source,
                                                                                                             train_target,
                                                                                                             BATCH_SIZE,
                                                                                                             words_to_index['<PAD>']))):
            _, loss = sess.run([train_op, cost],
                                  {inputs: source_batch,
                                   targets: target_batch,
                                   lr: LEARNING_RATE,
                                   target_sequence_length: targets_lengths,
                                   keep_prob: KEEP_PROB,
                                   source_sequence_length: sources_lengths})
    
            if batch_i % 1 == 0 and batch_i > 0:
                  batch_train_logits = sess.run(
                      inference_logits,
                      {inputs: source_batch,
                       target_sequence_length: targets_lengths,
                       keep_prob: 1.0,
                       source_sequence_length: sources_lengths})
    
                  batch_valid_logits = sess.run(
                      inference_logits,
                      {inputs: valid_sources_batch,
                       target_sequence_length: valid_targets_lengths,
                       keep_prob: 1.0,
                       source_sequence_length: valid_sources_lengths})
    
                  train_acc = get_accuracy(target_batch, batch_train_logits)
                  valid_acc = get_accuracy(valid_targets_batch, batch_valid_logits)       
    
    
                  print('Epoch {:>3} Batch {:>4}/{} - Train Accuracy: {:>6.4f}, Validation Accuracy: {:>6.4f}, Loss: {:>6.4f}'
                        .format(epoch, batch_i, len(article_list) // BATCH_SIZE, train_acc, valid_acc, loss))
            if batch_i%1 == 0 and batch_i > 0:
                  idx = np.random.randint(0,5)
                  st = sentence_to_seq(sentances[idx], words_to_index)
                
                  trans_logits = sess.run(inference_logits, feed_dict={inputs: [st]*BATCH_SIZE,
                                       target_sequence_length: [len(st)]*BATCH_SIZE,
                                       source_sequence_length: [len(st)]*BATCH_SIZE,
                                       keep_prob: KEEP_PROB})[0]
                  cst = sess.run(merged,{inputs: source_batch,
                                   targets: target_batch,
                                   lr: LEARNING_RATE,
                                   target_sequence_length: targets_lengths,
                                   source_sequence_length: sources_lengths,
                                   keep_prob: KEEP_PROB})
                  writer.add_summary(cst, batch_i)                
                
                  print('Input')
                  print('  Word Ids:      {}'.format([i for i in st]))
                  print('  input_article : {}'.format([index_to_words[i] for i in st]))
                
                  print('\nPrediction')
                  print('  Word Ids:      {}'.format([i for i in trans_logits]))
                  print('  output_title: {}'.format(" ".join([index_to_words[i] for i in trans_logits])))
    
        LEARNING_RATE *= LEARNING_RATE_DECAY
        if LEARNING_RATE < MIN_LR:
            LEARNING_RATE = MIN_LR
            # Save Model
        saver.save(sess, 'checkpoints/dev',epoch)
        print('Model Trained and Saved')