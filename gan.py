"""Simple GAN implementation."""

import tensorflow as tf


class GAN(tf.keras.Model):
    """Generative adversarial network.

    This is a simple but generic GAN class with two training schemes:

    1.  Alternate between training the discriminator and the generator
        separately.
    2.  Train the discriminator and generator simultaneously.

    Parameters
    ----------
    generator : tf.keras.Model instance
        The generator network. This network should accept one input tensor of
        shape (batch, latent_dim) drawn from the latent distribution and output
        a simulated example of arbitrary shape.

    discriminator : tf.keras.Model instance
        The discriminator network. This network should accept one inout tensor
        of the same shape as the output of the generator network and return a
        logit (not a probability) classifying whether the input is a real or
        fake example.

    alternate_training : bool, optional (default=True)
        Which of the two training schemes to use.

    rng : callable, optional (default=None)
        Function with signature (shape, seed=None, ...) -> tf.Tensor used to
        draw random samples from the latent distribution. The default is the
        standard normal distribution.
    """

    def __init__(self, generator, discriminator, alternate_training=True,
                 rng=None):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.alternate_training = alternate_training
        if rng is None:
            rng = tf.random.normal
        self.rng = rng

    def train_step(self, data):
        """One training step for the GAN.

        Parameters
        ----------
        data : nested structure of tf.Tensors
            Real examples for the discriminator.
        """
        # Unpack the input passed in by the fit() method
        inputs, *_ = tf.nest.flatten(data)

        # Trainable variables of the GAN
        disc_vars = self.discriminator.trainable_variables
        gen_vars = self.generator.trainable_variables

        loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        # Draw a sample from the latent distribution
        batch_size = tf.shape(inputs)[:1]
        latent_size = tf.nest.flatten(self.generator.inputs)[0].shape[1:]
        noise_shape = tf.concat([batch_size, latent_size], axis=0)
        noise = self.rng(noise_shape)

        if self.alternate_training:
            # Train the discriminator
            fake_inputs = self.generator(noise, training=False)
            with tf.GradientTape() as disc_tape:
                real_logits = self.discriminator(inputs, training=True)
                fake_logits = self.discriminator(fake_inputs, training=True)
                disc_loss = loss_fn(tf.ones_like(real_logits), real_logits)
                disc_loss += loss_fn(tf.zeros_like(fake_logits), fake_logits)
            disc_grads = disc_tape.gradient(disc_loss, disc_vars)
            self.optimizer.apply_gradients(zip(disc_grads, disc_vars))

            # Train the generator
            noise = self.rng(noise_shape)
            with tf.GradientTape() as gen_tape:
                fake_inputs = self.generator(noise, training=True)
                fake_logits = self.discriminator(fake_inputs, training=False)
                gen_loss = loss_fn(tf.ones_like(fake_logits), fake_logits)
            gen_grads = gen_tape.gradient(gen_loss, gen_vars)
            self.optimizer.apply_gradients(zip(gen_grads, gen_vars))
        else:
            # Train generator and discriminator simultaneously
            with tf.GradientTape() as disc_tape, tf.GradientTape() as gen_tape:
                fake_inputs = self.generator(noise, training=True)
                real_logits = self.discriminator(inputs, training=True)
                fake_logits = self.discriminator(fake_inputs, training=True)
                disc_loss = loss_fn(tf.ones_like(real_logits), real_logits)
                disc_loss += loss_fn(tf.zeros_like(fake_logits), fake_logits)
                gen_loss = loss_fn(tf.ones_like(fake_logits), fake_logits)
            disc_grads = disc_tape.gradient(disc_loss, disc_vars)
            gen_grads = gen_tape.gradient(gen_loss, gen_vars)
            self.optimizer.apply_gradients(zip(disc_grads, disc_vars))
            self.optimizer.apply_gradients(zip(gen_grads, gen_vars))

        return {'gen_loss': gen_loss, 'disc_loss': disc_loss}
