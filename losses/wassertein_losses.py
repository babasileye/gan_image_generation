import tensorflow as tf

def discriminator_loss(real_output,fake_output):
   """
   """
   real_loss=tf.reduce_mean(tf.math.multiply(-tf.ones_like(real_output,dtype=tf.float32),real_output))
   fake_loss=tf.reduce_mean(tf.math.multiply(tf.ones_like(fake_output,dtype=tf.float32),fake_output))
   total_loss=0.5*(real_loss+fake_loss)
   return total_loss

def generator_loss(fake_output):
   """
   """
   return tf.reduce_mean(tf.math.multiply(-tf.ones_like(fake_output,dtype=tf.float32), fake_output))
