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

#implement model w/ given equation
y = tf.nn.softmax(tf.matmul(x, W) + b)

#outgoing vals?
y_ = tf.placeholder(tf.float32, [None, 1])

#backprop
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

#train
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

#iterative train
for i in range(5):
  batch_xs, batch_ys = goData.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  print(i*100)
  #print(W.eval(sess))
  #print(b.eval(sess))
#a=tf.matmul(tf.Variable(tf.random_normal([1, 81])), W)
#e= sess.run(a)
tensArr= W.eval(sess).transpose()
consArr= b.eval(sess)
boardArr= np.array([0,0,0,0,0,0,0,0,0,
		    0,0,0,0,0,0,0,0,0,
		    0,0,0,0,1,1,0,1,0,
		    0,0,0,1,0,-1,0,0,0,
		    0,0,-1,0,0,0,-1,0,0,
		    0,0,0,-1,0,0,0,0,0,
		    0,0,1,0,0,-1,0,0,0,
		    0,0,0,0,0,0,0,0,0,
		    0,0,0,0,0,0,0,0,0])#1= black
highestOutcome= (np.dot(tensArr, boardArr)+consArr)
maxMove= -1

for count in range(0,81):# 
	if(boardArr[count]==0):	
		boardArr[count]= 1 #change
		temp= (np.dot(tensArr, boardArr)+consArr)
		if(temp>highestOutcome):#change to less
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

lowestOutcome= 0
minMove= -1

for count in range(0,81):
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
	
