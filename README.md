# `tf.keras` GAN

This repo contains a simple example of how to implement a generative adversarial network (GAN), including training logic, entirely within a `tf.keras` model.
Starting with TensorFlow 2.2, Keras models expose new `train_step`, `test_step`, and `predict_step` methods that allow customized behavior during training, evaluation, and prediction stages.
Previously, such custom behavior would have to be implemented outside a Keras model; now it can be implemented within.

See the [accompanying blog post](https://artemmavrin.github.io/blog/2020/03/tf-keras-custom-training-logic) and the [example Jupyter notebook](MNIST%20GAN.ipynb).

![](images/gan_animation.gif)
