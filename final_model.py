from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np
import pandas as pd
from scipy import io
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import numbers
from tensorflow.contrib import layers
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.layers import utils


# load data

y_tr = pd.read_csv('./tox21/tox21_labels_train.csv.gz', index_col=0, compression="gzip")
y_te = pd.read_csv('./tox21/tox21_labels_test.csv.gz', index_col=0, compression="gzip") 
x_tr_dense = pd.read_csv('./tox21/tox21_dense_train.csv.gz', index_col=0, compression="gzip").values
x_te_dense = pd.read_csv('./tox21/tox21_dense_test.csv.gz', index_col=0, compression="gzip").values
x_tr_sparse = io.mmread('./tox21/tox21_sparse_train.mtx.gz').tocsc()
x_te_sparse = io.mmread('./tox21/tox21_sparse_test.mtx.gz').tocsc()


# filter out very sparse features
sparse_col_idx = ((x_tr_sparse > 0).mean(0) > 0.05).A.ravel()
x_tr = np.hstack([x_tr_dense, x_tr_sparse[:, sparse_col_idx].A])
x_te = np.hstack([x_te_dense, x_te_sparse[:, sparse_col_idx].A])

#print x_tr_dense.columns
#print y_tr.columns
#print x_tr_sparse.columns


# Network Parameters
n_hidden_1 = 150# 1st layer number of features
n_hidden_2 = 150 # 2nd layer number of features
n_hidden_3 = 150
n_hidden_4 = 150
n_classes = 2
target="SR.ARE"

rows_tr = np.isfinite(y_tr[target]).values
rows_te = np.isfinite(y_te[target]).values
x_tr=x_tr_dense[rows_tr]
y_tr=y_tr[target][rows_tr]
y_tr=pd.get_dummies(y_tr)
x_te=x_te_dense[rows_te]
y_te=y_te[target][rows_te]
y_te=pd.get_dummies(y_te)
learning_rate = 0.05
training_epochs = 10
batch_size = 100
display_step = 1
n_input = x_tr.shape[1] 
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])
dropoutRate = tf.placeholder(tf.float32)
is_training= tf.placeholder(tf.bool)



#Definition of scaled exponential linear units (SELUs)


def selu(x):
    with ops.name_scope('selu') as scope:
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale*tf.where(x>=0.0, x, alpha*tf.nn.elu(x))

n = x_tr.shape[0]
batches_train_x = [x_tr[k:k+batch_size] for k in xrange(0, n,batch_size)]
batches_train_y = [y_tr[k:k+batch_size] for k in xrange(0, n,batch_size)]

n = x_te.shape[0]
batches_test_x = [x_te[k:k+batch_size] for k in xrange(0, n,batch_size)]
batches_test_y = [y_te[k:k+batch_size] for k in xrange(0, n,batch_size)]

#Definition of dropout variant for SNNs


def dropout_selu(x, rate, alpha= -1.7580993408473766, fixedPointMean=0.0, fixedPointVar=1.0, 
                 noise_shape=None, seed=None, name=None, training=False):
    """Dropout to a value with rescaling."""

    def dropout_selu_impl(x, rate, alpha, noise_shape, seed, name):
        keep_prob = 1.0 - rate
        x = ops.convert_to_tensor(x, name="x")
        if isinstance(keep_prob, numbers.Real) and not 0 < keep_prob <= 1:
            raise ValueError("keep_prob must be a scalar tensor or a float in the "
                                             "range (0, 1], got %g" % keep_prob)
        keep_prob = ops.convert_to_tensor(keep_prob, dtype=x.dtype, name="keep_prob")
        keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())

        alpha = ops.convert_to_tensor(alpha, dtype=x.dtype, name="alpha")
        keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())

        if tensor_util.constant_value(keep_prob) == 1:
            return x

        noise_shape = noise_shape if noise_shape is not None else array_ops.shape(x)
        random_tensor = keep_prob
        random_tensor += random_ops.random_uniform(noise_shape, seed=seed, dtype=x.dtype)
        binary_tensor = math_ops.floor(random_tensor)
        ret = x * binary_tensor + alpha * (1-binary_tensor)

        a = tf.sqrt(fixedPointVar / (keep_prob *((1-keep_prob) * tf.pow(alpha-fixedPointMean,2) + fixedPointVar)))

        b = fixedPointMean - a * (keep_prob * fixedPointMean + (1 - keep_prob) * alpha)
        ret = a * ret + b
        ret.set_shape(x.get_shape())
        return ret

    with ops.name_scope(name, "dropout", [x]) as name:
        return utils.smart_cond(training,
            lambda: dropout_selu_impl(x, rate, alpha, noise_shape, seed, name),
            lambda: array_ops.identity(x))



# Scale input to zero mean and unit variance
scaler = StandardScaler().fit(x_tr)



# Tensorboard
logs_path = './tmp'



# Create model
def multilayer_perceptron(x, weights, biases, rate, is_training):
    # Hidden layer with SELU activation
    #print (x.shape)
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    #print (layer_1.get_shape)
    layer_1 = selu(layer_1)
    layer_1 = dropout_selu(layer_1,rate, training=is_training)
    
    # Hidden layer with SELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    #print (layer_2.get_shape)
    layer_2 = selu(layer_2)
    layer_2 = dropout_selu(layer_2,rate, training=is_training)

    # Hidden layer with SELU activation
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    #print (layer_2.get_shape)
    layer_3 = selu(layer_3)
    layer_3 = dropout_selu(layer_3,rate, training=is_training)

    # Hidden layer with SELU activation
    layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
    #print (layer_2.get_shape)
    layer_4 = selu(layer_4)
    layer_4 = dropout_selu(layer_4,rate, training=is_training)

    # Output layer with linear activation
    out_layer = tf.matmul(layer_4, weights['out']) + biases['out']
    out_layer=tf.nn.softmax(out_layer)
    #print (out_layer.get_shape)
    return out_layer



# Store layers weight & bias


weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1],stddev=np.sqrt(1/n_input))),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2],stddev=np.sqrt(1/n_hidden_1))),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3],stddev=np.sqrt(1/n_hidden_2))),
    'h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4],stddev=np.sqrt(1/n_hidden_3))),   
    'out': tf.Variable(tf.random_normal([n_hidden_4, n_classes],stddev=np.sqrt(1/n_hidden_4)))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1],stddev=0)),
    'b2': tf.Variable(tf.random_normal([n_hidden_2],stddev=0)),
    'b3': tf.Variable(tf.random_normal([n_hidden_3],stddev=0)),
    'b4': tf.Variable(tf.random_normal([n_hidden_4],stddev=0)),
    'out': tf.Variable(tf.random_normal([n_classes],stddev=0))
}
# Construct model
pred = multilayer_perceptron(x, weights, biases, rate=dropoutRate, is_training=is_training)


# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

 # Test model
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
a = tf.reduce_max(pred, reduction_indices=[1])
#a=tf.reduce_max(y, reduction_indices=[1])
b=y_te #tf.reduce_max(y_te, reduction_indices=[1])

         
# Initializing the variables
init = tf.global_variables_initializer()



# Create a histogramm for weights
tf.summary.histogram("weights2", weights['h2'])
tf.summary.histogram("weights1", weights['h1'])

# Create a summary to monitor cost tensor
tf.summary.scalar("loss", cost)
# Create a summary to monitor accuracy tensor
tf.summary.scalar("accuracy", accuracy)
# Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()



# Launch the graph
gpu_options = tf.GPUOptions(allow_growth=True)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(init)

    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(x_tr.shape[0]/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = batches_train_x[i],batches_train_y[i]
            batch_x = scaler.transform(batch_x)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y, dropoutRate: 0.05, is_training:True})

            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print ("Epoch:", '%04d' % (epoch+1), "cost=","{:.9f}".format(avg_cost))
            
            accTrain, costTrain, summary = sess.run([accuracy, cost, merged_summary_op], 
                                                        feed_dict={x: batch_x, y: batch_y, 
                                                                   dropoutRate: 0.0, is_training:False})
            summary_writer.add_summary(summary, epoch)
            
            print("Train-Accuracy:", accTrain,"Train-Loss:", costTrain)

            batch_x_test, batch_y_test = x_te,y_te #batches_test_x[i],batches_test_y[i]
            batch_x_test = scaler.transform(batch_x_test)
            accTest, costVal = sess.run([accuracy, cost], feed_dict={x: batch_x_test, y: batch_y_test, 
                                                                   dropoutRate: 0.0, is_training:False})

            sess.run(tf.local_variables_initializer())

            pred_score= sess.run(pred,feed_dict={x: batch_x_test, y: batch_y_test,dropoutRate: 0.0, is_training:False})

            sklearn_auc = roc_auc_score(y_true=b,y_score=pred_score)

            print("Validation-AUC:", sklearn_auc,"Val-Loss:", costVal,"\n")

