import tensorflow as tf
import numpy as np
import math

## SET BACKEND
import matplotlib as mpl
mpl.use('TkAgg')

import matplotlib.pyplot as plt
import matplotlib.animation as animation

# generate house size
num_house = 160
np.random.seed(42)
house_size = np.random.randint(low=1000, high=3500, size=num_house)

#generate house prices
np.random.seed(42)
house_price = house_size * 100.0 + np.random.randint(low=20000, high=70000, size=num_house)

#Plot generated house and size
plt.plot(house_size, house_price, "bx") #bx = blue
plt.ylabel("Price")
plt.xlabel("Size")
plt.show()

#normalize data
def normalize(array):
    return(array - array.mean()) / array.std()

#define number of training samples, we take 70%. 
# We can take the first 70% because its random
num_train_samples = math.floor(num_house * 0.7)

#define training data
train_house_size = np.asarray(house_size[:num_train_samples])
train_price = np.asarray(house_price[:num_train_samples])

train_house_size_norm = normalize(train_house_size)
train_price_norm = normalize(train_price)

#define test data
test_house_size = np.asarray(house_size[num_train_samples:])
test_price = np.asarray(house_price[num_train_samples:])

test_house_size_norm = normalize(test_house_size)
test_price_norm = normalize(test_price)

#set up tensor flow placeholders 
tf_house_size = tf.placeholder("float", name="house_size")
tf_price = tf.placeholder("float", name="price")

#define the variables holding the size_factor and price we set during training
#we initialize them to some random values based on the normal distribution
tf_size_factor = tf.Variable(np.random.randn(), name="size_factor")
tf_price_offset = tf.Variable(np.random.randn(), name="price_offset")

# define the operations for the predicting values : predicted price = (size_factor * house_size) + price_offset
# notice the use of the tensorflow add and multiply functions. The add the operations to the computation graph,
# and the tensorflow methods understand how to deals with Tensors and dont try to use other libray methods
tf_price_prediction = tf.add(tf.multiply(tf_size_factor, tf_house_size), tf_price_offset)

#define the loss function - mean squared error
tf_cost = tf.reduce_sum(tf.pow(tf_price_prediction-tf_price, 2))/(2*num_train_samples)

#optimizer learning rate. The size of the steps down the gradient
learning_rate = 0.1

#optimize with a gradient descent optimizer that will minimize the loss defined above
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(tf_cost)

#initializing the variables
init = tf.global_variables_initializer()

#launch the graph in the session
with tf.Session() as sess:
    sess.run(init)

    #set how often to display training progress and number of training iterations
    display_every = 2
    num_training_iter = 50
    training_cost = 0
    
    #keep iterating the training data
    for iteration in range(num_training_iter):
    

        #fit all training data
        for(x,y) in zip(train_house_size_norm, train_price_norm):
            sess.run(optimizer, feed_dict={tf_house_size: x, tf_price: y})

        #display current status
        if(iteration + 1) % display_every == 0:
            training_cost = sess.run(tf_cost, feed_dict={tf_house_size: train_house_size_norm, tf_price: train_price_norm})
            print("iteration #:", '%04d' % (iteration + 1), "cost=", "{:.9f}".format(training_cost), \
                "size_factor=", sess.run(tf_size_factor), "price_offset=", sess.run(tf_price_offset))

    print("optimization finished!")
    print("Trained cost=", training_cost, "size_factor=", sess.run(tf_size_factor), "price_offset=", sess.run(tf_price_offset), '\n')


   # Plot of training and test data, and learned regression
    
    # get values used to normalized data so we can denormalize data back to its original scale
    train_house_size_mean = train_house_size.mean()
    train_house_size_std = train_house_size.std()

    train_price_mean = train_price.mean()
    train_price_std = train_price.std()

    # Plot the graph
    plt.rcParams["figure.figsize"] = (10,8)
    plt.figure()
    plt.ylabel("Price")
    plt.xlabel("Size (sq.ft)")
    plt.plot(train_house_size, train_price, 'go', label='Training data')
    plt.plot(test_house_size, test_price, 'mo', label='Testing data')
    plt.plot(train_house_size_norm * train_house_size_std + train_house_size_mean,
             (sess.run(tf_size_factor) * train_house_size_norm + sess.run(tf_price_offset)) * train_price_std + train_price_mean,
             label='Learned Regression')
 
    plt.legend(loc='upper left')
    plt.show()
