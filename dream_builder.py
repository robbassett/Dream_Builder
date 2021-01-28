import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K

import numpy as np
import imageio
import matplotlib.pyplot as plt
from scipy.optimize import fmin_l_bfgs_b
import glob
import time
import IPython.display as display
import PIL.Image

def build_dreamer(mod_in):
    mod_out = keras.Sequential()
    l1_config = mod_in.layers[0].get_config()
    new_config = {}
    for k in l1_config.keys():
        if k != 'batch_input_shape':
            new_config[k] = l1_config[k]
        else:
            new_config['input_shape'] = (None,None,l1_config[k][-1])
            
    mod_out.add(layers.Conv2D(**new_config))
    for layer in mod_in.layers[1:]:
        if isinstance(layer, layers.Flatten):
            break
        ltype = layer.__class__
        mod_out.add(ltype(**layer.get_config()))
    return mod_out

class DeepDream(tf.Module):
  def __init__(self, model):
    self.model = model

  @tf.function(
      input_signature=(
        tf.TensorSpec(shape=[None,None,3], dtype=tf.float32),
        tf.TensorSpec(shape=[], dtype=tf.int32),
        tf.TensorSpec(shape=[], dtype=tf.float32),)
  )
  def __call__(self, img, steps, step_size):
      print("Tracing")
      loss = tf.constant(0.0)
      for n in tf.range(steps):
        with tf.GradientTape() as tape:
          # This needs gradients relative to `img`
          # `GradientTape` only watches `tf.Variable`s by default
          tape.watch(img)
          loss = calc_loss(img, self.model)

        # Calculate the gradient of the loss with respect to the pixels of the input image.
        gradients = tape.gradient(loss, img)

        # Normalize the gradients.
        gradients /= tf.math.reduce_std(gradients) + 1e-8 

        # In gradient ascent, the "loss" is maximized so that the input image increasingly "excites" the layers.
        # You can update the image by directly adding the gradients (because they're the same shape!)
        img = img + gradients*step_size
        img = tf.clip_by_value(img, -1, 1)

      return loss, img

def run_deep_dream_simple(img, deepdream, steps=100, step_size=0.01):
  # Convert from uint8 to the range expected by the model.
  img = tf.keras.applications.inception_v3.preprocess_input(img)
  img = tf.convert_to_tensor(img)
  step_size = tf.convert_to_tensor(step_size)
  steps_remaining = steps
  step = 0
  while steps_remaining:
    if steps_remaining>100000:
      run_steps = tf.constant(100)
    else:
      run_steps = tf.constant(steps_remaining)
    steps_remaining -= run_steps
    step += run_steps

    loss, img = deepdream(img, run_steps, tf.constant(step_size))

    print ("Step {}, loss {}".format(step, loss))

  result = deprocess(img)

  return result

def calc_loss(img, model):
  # Pass forward the image through the model to retrieve the activations.
  # Converts the image into a batch of size 1.
  img_batch = tf.expand_dims(img, axis=0)
  layer_activations = model(img_batch)
  if len(layer_activations) == 1:
    layer_activations = [layer_activations]

  losses = []
  for act in layer_activations:
    loss = tf.math.reduce_mean(act)
    losses.append(loss)

  return  tf.reduce_sum(losses)

def deprocess(img):
  img = 255*(img + 1.0)/2.0
  return tf.cast(img, tf.uint8)

def DreamMyImage(image_file,model,weights_file,output_name,layer_names=[],nlayer=4,octaves=True,octave_scale=1.4,nstep=10,step_size=0.035):

    dreamer = build_dreamer(model)
    dreamer.load_weights(weights_file)

    if len(layer_names) == 0:
        names = []
        for layer in dreamer.layers: names.append(layer.name)
        names = np.array(names)
        names = np.random.choice(names,4,replace=False)
    else:
        names = layer_names
        
    lyrs = [dreamer.get_layer(name).output for name in names]

    dream_model = tf.keras.Model(inputs=dreamer.input, outputs=lyrs)
    deepdream = DeepDream(dream_model)

    img = imageio.imread(image_file)
    base_shape = tf.shape(img)[:-1]
    if octaves:
        float_base_shape = tf.cast(base_shape, tf.float32)
        for n in range(-2, 3):
            new_shape = tf.cast(float_base_shape*(octave_scale**n), tf.int32)
            img = tf.image.resize(img, new_shape).numpy()
            img = run_deep_dream_simple(img,deepdream,steps=nstep,step_size=step_size)
    else:
        img = run_deep_dream_simple(img,deepdream,steps=nstep,step_size=step_size)
            
    imageio.imsave(output_name,img)
