import numpy as np
import tensorflow as tf

class Model(object):

  def __init__(self, num_classes, seq_len, word_dict_size, wv, num_filters1=150, num_filters2 = 100, filter1_size=2, filter2_size=5, emb_size=100, dist_emb_size=10, l2_reg_lambda = 0.01):

    tf.reset_default_graph()
    # model_path = './ckpt/lstm-cnn-att/model.ckpt'
    
    self.w  = tf.placeholder(tf.int32, [None, seq_len], name="x")
    self.input_y = tf.placeholder(tf.int32, [None, num_classes], name="input_y")
    self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

    # Initialization
    W_emb = tf.Variable(wv,name='W_emb',trainable=True)
    # W_emb = tf.Variable(tf.truncated_normal([len(wv),emb_size], stddev=0.1),trainable=True,name='W_emb')
    
    # Embedding layer
    X = tf.nn.embedding_lookup(W_emb, self.w)
    
   
    
    # CNN+Maxpooling Layer
    with tf.variable_scope('cnn'):
      conv_out = tf.layers.conv1d(X, num_filters2, 3, strides=1, padding='same')
      conv_out = tf.nn.relu(conv_out)
      
      ##Maxpooling
      pool_out = tf.layers.max_pooling1d(conv_out, seq_len, seq_len, padding='same')
      h2_pool = tf.squeeze(pool_out, axis=1)
    
    ##Dropout
    h_flat = tf.reshape(h2_pool,[-1,num_filters2])
    # h_flat = tf.reshape(h2_cnn,[-1,(seq_len-3*(filter_size-1))*2*num_filters])
    h_drop = tf.nn.dropout(h_flat,self.dropout_keep_prob)

    # Fully connetected layer
    W = tf.Variable(tf.truncated_normal([num_filters2, num_classes], stddev=0.1), name="W")
    # W = tf.Variable(tf.truncated_normal([(seq_len-3*(filter_size-1))*2*num_filters, num_classes], stddev=0.1), name="W")
    b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
    scores = tf.nn.xw_plus_b(h_drop, W, b, name="scores")
    

    l2_loss = tf.constant(0.0)
    l2_loss += tf.nn.l2_loss(W)


    # prediction and loss function
    self.predictions = tf.argmax(scores, 1)
    self.losses = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=self.input_y)
    self.loss = tf.reduce_mean(self.losses) + l2_reg_lambda*l2_loss

    # Accuracy
    self.correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
    self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, "float"))  
    self.global_step = tf.Variable(0,name='global_step',trainable=False)
    self.optimizer = tf.train.AdamOptimizer(1e-3).minimize(self.loss,global_step=self.global_step)

    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    self.sess = tf.Session(config=session_conf)
    
    self.sess.run(tf.global_variables_initializer())



  def train_step(self, W_batch, y_batch):
      feed_dict = {
        self.w     :W_batch,
        self.dropout_keep_prob: 0.7,
        self.input_y   :y_batch
          }
      _, step, loss, accuracy, predictions = self.sess.run([self.optimizer, self.global_step, self.loss, self.accuracy, self.predictions], feed_dict)
      # print ("step "+str(step) + " loss "+str(loss) +" accuracy "+str(accuracy))
      return step,accuracy

  def test_step(self, W_batch, d1_batch, d2_batch, y_batch):
      feed_dict = {
        self.w     :W_batch,
        self.dropout_keep_prob:1.0,
        self.input_y :y_batch
        }
      loss, accuracy, predictions = self.sess.run([self.loss, self.accuracy, self.predictions], feed_dict)
      return predictions

train_file = 'SemEval2010Task8/train.cln'
test_file = 'SemEval2010Task8/test.cln'

def load_data(filename):
  W = []
  Y = []
  with open(filename) as file:
    for line in file:
      segments = line.strip().split()
      label = int(segments[0])
      onehot = np.zeros(19)
      onehot[label] = 1.0
      Y.append(onehot)
      W.append(segments[5:])
  return W, Y

W_train, Y_train = load_data(train_file)
W_test, Y_test = load_data(test_file)

vocab = set()
seq_len = 0
for sentence in W_train + W_test:
  n = len(sentence)
  if seq_len < n:
    seq_len = n
  for token in sentence:
    vocab.add(token)

vocab2id = {'<pad>':0}
for i, token in enumerate(vocab):
  vocab2id[token] = i+1

for i in range(len(W_train)):
  W_train[i] = [vocab2id[tok] for tok in W_train[i]]
  pad_n = seq_len - len(W_train[i])
  W_train[i].extend([0]*pad_n)

for i in range(len(W_test)):
  W_test[i] = [vocab2id[tok] for tok in W_test[i]]
  pad_n = seq_len - len(W_test[i])
  W_test[i].extend([0]*pad_n)

num_epochs = 30
batch_size = 100
num_batches_per_epoch = int( len(W_train)/batch_size )

n_vocab = len(vocab2id)
word_dim = 100
wv = np.random.normal(scale=word_dim**-0.5, size=[n_vocab, word_dim])
wv = wv.astype(np.float32)

model = Model(19,seq_len, n_vocab,wv)

for j in range(num_epochs):
  acc = []
  step = 0
  sam=[]
  for batch_num in range(num_batches_per_epoch):  
    start_index = batch_num*batch_size
    end_index = (batch_num + 1) * batch_size
    sam.append(slice(start_index, end_index))    
  
  for rang in sam:
    step,acc_cur  = model.train_step(W_train[rang], Y_train[rang]) 
    acc.append(acc_cur)
  train_acc = np.mean(np.array(acc))
  
  pred = model.test_step(W_test, None, None, Y_test)
  y_true = np.argmax(Y_test, 1)
  y_pred = pred
  test_acc = np.mean(np.equal(y_true, y_pred))

  
  print("epoch %d, train acc %.4f, test acc %.4f" % (j+1, train_acc, test_acc))

# epoch 1, train acc 0.1654, test acc 0.1999
# epoch 2, train acc 0.2946, test acc 0.2985
# epoch 3, train acc 0.3654, test acc 0.3397
# epoch 4, train acc 0.4224, test acc 0.4034
# epoch 5, train acc 0.5001, test acc 0.4446
# epoch 6, train acc 0.5851, test acc 0.4755
# epoch 7, train acc 0.6667, test acc 0.5035
# epoch 8, train acc 0.7425, test acc 0.5259
# epoch 9, train acc 0.8097, test acc 0.5502
# epoch 10, train acc 0.8655, test acc 0.5679
# epoch 11, train acc 0.9014, test acc 0.5778
# epoch 12, train acc 0.9269, test acc 0.5834
# epoch 13, train acc 0.9451, test acc 0.5896
# epoch 14, train acc 0.9614, test acc 0.5915
# epoch 15, train acc 0.9679, test acc 0.5878
# epoch 16, train acc 0.9760, test acc 0.5856
# epoch 17, train acc 0.9799, test acc 0.5881









