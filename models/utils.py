import matplotlib.pyplot as plt 
import tensorflow as tf 

from tensorflow.keras.constraints import Constraint

# clip model weights to a given hypercube
class ClipConstraint(Constraint):
   # set clip value when initialized
   def __init__(self, clip_value):
      self.clip_value = clip_value

   # clip model weights to hypercube
   def __call__(self, weights):
      return tf.clip_by_value(weights, -self.clip_value, self.clip_value)

   # get the config
   def get_config(self):
      return {'clip_value': self.clip_value}
      
def generate_and_save_images(generator,test_input,image_name,output_dir):
   # Notice `training` is set to False.
   # This is so all layers run in inference mode (batchnorm).
   predictions = generator(test_input, training=False)
   fig = plt.figure(figsize=(7,7))
   for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow((1.0+np.squeeze(predictions[i]))/2.0,cmap='gray')
      plt.axis('off')
   plt.savefig(f'{output_dir}/{image_name}.png')
   plt.show()
