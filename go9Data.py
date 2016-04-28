
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import tensorflow.python.platform
import numpy
import csv
import pickle
import binascii
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
def pullPositions(filename):
  #Extract the images into a 4D uint8 numpy array [index, y, x, depth].
  
  with open(filename, 'r') as f:
    count=0
    domain=[]
    rang=[]
    print('Extracting', filename)
    while(count<850000):#f):
	if(count%1000==0):
		print(count)
	count+=1
	zobrist=f.read(8)
	f.read(1)
	board= []
	sign=1
	playerTurn= f.read(1)
	if playerTurn=='W': 
		sign=-1
	else: 
		sign = 1
	for c in f.read(81):
		if c== '.': board.append(0.0)		
		elif c== '#': board.append(1.0)		
		elif c== 'O': board.append(-1.0)
		else: board.append(c)
	zts= zobristToScore(zobrist)
	if(zts!= None):
		domain.append(board)
		rang.append(sign*int(binascii.hexlify(zts[0]),16)*int(binascii.hexlify(zts[1]),16))
	if(f.read(1)== hex(1)): f.read(8)
    pickle.dump(domain,open("domain3.p", "wb"))
    pickle.dump(rang,open("range3.p", "wb"))
  '''
  domain= pickle.load(open("domain2.p", "rb"))

  rang= pickle.load(open("range2.p", "rb"))
  '''
  M= max(rang)
  m= min(rang)
  dc=0
  '''
  for x in range(0,9):
  	print(domain[rang.index(M)][x:x+9])

  print(m)
  for x in range(0,9):
  	print(domain[rang.index(m)][x:x+9])
  print()
  for x in range(0,9):
  	print(domain[-1][x:x+9])
  print(rang[-1])

  for x in range(0,9):
  	print(domain[10][x:x+9])
  #rang[1000]  
  '''
  print("pullPositions")
  domain= numpy.array(domain)
  rang= numpy.array(rang)
  return domain,rang

class DataSet(object):
  def __init__(self, goBoards, labels, dtype=tf.float32):
    """Construct a DataSet.
    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.
    """
    assert goBoards.shape[0] == labels.shape[0], ('images.shape: %s labels.shape: %s' % (goBoards.shape,
                                                 labels.shape))
    self._num_examples = goBoards.shape[0]
    # Convert shape from [num examples, rows, columns, depth]
    # to [num examples, rows*columns] (assuming depth == 1)
    #assert goBoards.shape[3] == 1
    goBoards = goBoards.reshape(goBoards.shape[0],81)
    labels = labels.reshape(labels.shape[0],1)
    if dtype == tf.float32:
	goBoards = goBoards.astype(numpy.int8)
    dtype = tf.as_dtype(dtype).base_dtype
    self._goBoards = goBoards
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0
    self._num_examples = 50000#images.shape[0]
  @property
  def goBoards(self):
    return self._goBoards
  @property
  def labels(self):
    return self._labels
  @property
  def num_examples(self):
    return self._num_examples
  @property
  def epochs_completed(self):
    return self._epochs_completed
  def next_batch(self, batch_size, fake_data=False):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm)
      self._goBoards = self._goBoards[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._goBoards[start:end], self._labels[start:end]
def read_data_sets(dtype=tf.float32):#train_dir, ):
  class DataSets(object):
    pass
  data_sets = DataSets()
  pos_file = os.path.join("9x9/", "input_positions.dat")
  zobrist_file = os.path.join("9x9/", "zobrist_board_81.dat")
  temp=pullPositions(pos_file)
  dom= temp[0]
  ran= temp[1]
  #print(ran)
  #train_boards, train_labels = extract_boardAndLabel(local_file)
  data_sets.train = DataSet(dom[0:50000], ran[0:50000], dtype=dtype)
  return data_sets

def zobristToScore(zob):#dictionary
	with open(os.path.join("9x9/", "fuego_chinese.dat"), 'r') as f:
	   for line in f:
		if (zob in line and line.find(zob)+9< len(line)):#efficiency
			if(line[line.find(zob)+9]=='W' or line[line.find(zob)+9]=='B') :
				ind= line.find(zob)
				return line[ind+10:ind+12]

pos_file = os.path.join("9x9/", "input_positions.dat")
pullPositions(pos_file)
#binascii.hexlify(x))


