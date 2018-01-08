import source.crnn_max as ModelSource
import numpy as np
import tensorflow as tf


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

model = ModelSource.Model(19,seq_len, n_vocab,wv)

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

# epoch 1, train acc 0.1981, test acc 0.1807
# epoch 2, train acc 0.3914, test acc 0.3268
# epoch 3, train acc 0.5724, test acc 0.4052
# epoch 4, train acc 0.7391, test acc 0.4269
# epoch 5, train acc 0.8474, test acc 0.4222
# epoch 6, train acc 0.9075, test acc 0.4564
# epoch 7, train acc 0.9455, test acc 0.4873
# epoch 8, train acc 0.9702, test acc 0.4759
# epoch 9, train acc 0.9795, test acc 0.4678
# epoch 10, train acc 0.9888, test acc 0.4950
# epoch 11, train acc 0.9919, test acc 0.4763
# epoch 12, train acc 0.9945, test acc 0.4829
# epoch 13, train acc 0.9965, test acc 0.4833
# epoch 14, train acc 0.9964, test acc 0.4914
# epoch 15, train acc 0.9964, test acc 0.4928
# epoch 16, train acc 0.9959, test acc 0.4836
# epoch 17, train acc 0.9959, test acc 0.4766








