import go9Data
import tensorflow as tf
import numpy as np
goData = go9Data.read_data_sets()
sess = tf.InteractiveSession()

#flattened array containing floats
x = tf.placeholder(tf.float32, [None, 81])
#matrices of given dim filled with zeroes
W = tf.Variable(tf.random_normal([81, 1]))
b = tf.Variable(tf.random_normal([1]))

y_ = tf.placeholder(tf.float32, [None, 1])
#implement model w/ given equation
#y = tf.nn.softmax(tf.matmul(x, W) + b)

#backprop
#cross_entropy = -tf.reduce_sum(y_*tf.log(y))

#train
#train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

	

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
x_board = tf.reshape(x, [-1,9,9,1])#USE THE IMPORTANT VALUES
h_conv1 = tf.nn.relu(conv2d(x_board, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([3 * 3 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 3*3*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 1])
b_fc2 = bias_variable([1])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.initialize_all_variables())

for i in range(200):
  batch = goData.train.next_batch(50)
  if i%100 == 0:
    #train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
    print i#"step %d, training accuracy %g"%(i, train_accuracy)
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

tensArr= W.eval(sess).transpose()
consArr= b.eval(sess)
boardArr= np.array([0,0,0,0,0,0,0,0,0,
		    0,0,0,0,0,0,0,0,0,
		    0,0,0,0,0,0,0,0,0,
		    0,0,0,0,0,0,0,0,0,
		    0,0,0,0,0,0,0,0,0,
		    0,0,0,0,0,0,0,0,0,
		    0,0,0,0,0,0,0,0,0,
		    0,0,0,0,0,0,0,0,0,
		    0,0,0,0,0,0,0,0,0])#1= black
temp=[]
for i in range(0,len(boardArr)):
	temp.append(np.float32(boardArr[i]))
boardArr= temp
highestOutcome= 0
maxMove= -1

for count in range(0,81):# 
	if(boardArr[count]==0):	
		boardArr[count]= np.float32(1.0) #change
		x_board = tf.reshape(boardArr, [-1,9,9,1])
		h_conv1 = tf.nn.relu(conv2d(x_board, W_conv1) + b_conv1)
		h_pool1 = max_pool_2x2(h_conv1)	
		h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
		h_pool2 = max_pool_2x2(h_conv2)
		h_pool2_flat = tf.reshape(h_pool2, [-1, 3*3*64])
		h_fc1 = tf.nn.softmax(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
		temp= (tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2))
		if(temp<highestOutcome):#change to less
			highestOutcome= temp
			maxMove= count
		boardArr[count]= 0
print(maxMove)
print(highestOutcome)
boardArr[maxMove]= 1 #change
print(boardArr[0:9])
print(boardArr[9:18])
print(boardArr[18:27])
print(boardArr[27:36])
print(boardArr[36:45])
print(boardArr[45:54])
print(boardArr[54:63])
print(boardArr[63:72])
print(boardArr[72:81])
'''
lowestOutcome= 0
minMove= -1

for count in range(0,81):# of valid moves
	print(boardArr[count]==0)
	if(boardArr[count]==0):	
		boardArr[count]= -1 #change
		temp= (np.dot(tensArr, boardArr)+consArr)[0]
		if(temp<lowestOutcome):#change to less
			lowestOutcome= temp
			minMove= count
		boardArr[count]= 0
print(minMove)
print(lowestOutcome)
boardArr[minMove]= -1 #change
print(boardArr[0:9])
print(boardArr[9:18])
print(boardArr[18:27])
print(boardArr[27:36])
print(boardArr[36:45])
print(boardArr[45:54])
print(boardArr[54:63])
print(boardArr[63:72])
print(boardArr[72:81])
'''	
