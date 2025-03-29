import tensorflow as tf

# Run a simple operation on the GPU to verify
with tf.device('/GPU:0'):
    random_matrix = tf.random.normal([10000, 10000])
    matrix_multiply = tf.matmul(random_matrix, random_matrix)

print("GPU computation is working!")

