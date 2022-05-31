from model.srgan import generator, discriminator_new
import tensorflow as tf


discriminator = discriminator_new()
tf.keras.utils.plot_model(discriminator, show_shapes=True, dpi=64)
