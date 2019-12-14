import tensorflow as tf

X = tf.constant(0.5)
Y = tf.constant(0.0)
W = tf.Variable(1.0)

predict_Y = tf.multiply(X,W)

cost = tf.pow(Y - predict_Y,2)
min_cost = tf.train.GradientDescentOptimizer(0.025).minimize(cost)

for i in [X,W,Y,predict_Y,cost]:
    tf.summary.scalar(i.op.name,i)
    
summaries = tf.summary.merge_all()

with tf.Session() as s:
    summary_writer = tf.summary.FileWriter('single_input_neuron',s.graph)
    s.run(tf.global_variables_initializer())
    for i in range(100):
        summary_writer.add_summary(s.run(summaries),i)
        s.run(min_cost)
