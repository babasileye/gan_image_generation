import tensorflow as tf

from utils import ClipConstraint

def create_discriminator(image_shape=(28,28,1),layer_sizes=[64,128,256],dropout_rate=0.25,model_name='discriminator'):
   """
   """
   clip_constraint= ClipConstraint(0.01)
   
   model = tf.keras.Sequential(name='discriminator')
   for size in layer_sizes:
      model.add(tf.keras.layers.Conv2D(size,(3,3),strides=(2,2),padding='same',kernel_constraint=clip_constraint,input_shape=image_shape))
      model.add(tf.keras.layers.LeakyReLU())
   model.add(tf.keras.layers.Flatten())
   model.add(tf.keras.layers.Dense(1))
   return model

def create_generator():
   """
   """
   model = tf.keras.Sequential()
   model.add(tf.keras.layers.Dense(7*7*256,use_bias=False,input_shape=(100,)))
   model.add(tf.keras.layers.Reshape((7,7,256)))

   model.add(tf.keras.layers.Conv2DTranspose(128,kernel_size=3,strides=2,padding='same'))
   assert model.output_shape == (None, 14, 14, 128)
   model.add(tf.keras.layers.BatchNormalization())
   model.add(tf.keras.layers.LeakyReLU(alpha=0.01))
   
   model.add(tf.keras.layers.Conv2DTranspose(64,kernel_size=3,strides=1,padding='same'))
   assert model.output_shape == (None, 14, 14, 64)
   model.add(tf.keras.layers.BatchNormalization())
   model.add(tf.keras.layers.LeakyReLU(alpha=0.01))

   model.add(tf.keras.layers.Conv2DTranspose(1, kernel_size=3, strides=2, padding='same', use_bias=False, activation='tanh'))
   assert model.output_shape == (None, 28, 28, 1)

   return model
