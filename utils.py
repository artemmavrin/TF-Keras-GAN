"""Miscellaneous utility functions/classes."""

import pathlib

import tensorflow as tf
import matplotlib.pyplot as plt


class GenerateGANImages(tf.keras.callbacks.Callback):
    """Save GAN generated images after every training epoch.

    Parameters
    ----------
    pattern : str
        File pattern for images. Can use 'epoch' as a format argument (for
        example, pattern='image_{epoch:03d}.png').

    nrows : int, optional
        Number of rows in the generated image.

    ncols : int, optional
        Number of columns in the generated image.

    figsize : (int, int), optional
        Size of the image.

    seed : int, optional
        Seed for the GAN's latent distribution RNG.
    """
    def __init__(self, pattern, nrows=10, ncols=10, figsize=(10, 10),
                 seed=None):
        super().__init__()
        self.noise = None
        self.directory = pathlib.Path(pattern).parent
        self.pattern = pattern
        self.nrows = nrows
        self.ncols = ncols
        self.figsize = figsize
        self.seed = seed

    def _save_image(self, epoch):
        if self.noise is None:
            # Generate latent noise if it hasn't already been generated. We
            # reuse the same noise for each image to see how the generated
            # images change over time
            latent_dim = self.model.generator.input_shape[1]
            noise_shape = [self.nrows * self.ncols, latent_dim]
            self.noise = self.model.rng(noise_shape, seed=self.seed)
        images = self.model.generator(self.noise, training=False)

        fig, axs = plt.subplots(nrows=self.nrows, ncols=self.ncols,
                                figsize=self.figsize)

        for image, ax in zip(images, axs.flat):
            if image.shape[-1] == 1:
                # Remove single grayscale channel
                image = image[..., 0]
                cmap = 'binary'
            else:
                cmap = None
            ax.imshow(image, cmap=cmap, interpolation='none')
            ax.axis('off')
        fig.suptitle(f'Epoch {epoch}', fontsize=14)
        self.directory.mkdir(parents=True, exist_ok=True)
        plt.savefig(self.pattern.format(epoch=epoch), fontsize=14)
        plt.close()

    def on_train_begin(self, logs=None):
        self._save_image(epoch=0)

    def on_epoch_end(self, epoch, logs=None):
        self._save_image(epoch=epoch + 1)
