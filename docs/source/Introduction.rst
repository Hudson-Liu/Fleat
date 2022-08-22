*************************************************
Fleat: A Fast and Lightweight Learning Rate Tuner
*************************************************

FLEAT: **F**\ast **LEA**\rning rate **T**\uner

About Fleat
===========

Fleat is a fast and lightweight learning rate tuner for Keras models.

Traditional hyperparameter tuners suffer from their long runtimes and high resource consumption; even the best tuners (such as the Hyperband Tuner) require training the model 2 or 3 times over. 
Although not as generalizable as hyperparameter tuners, **Fleat makes learning rate optimization much faster and less resource intensive.** 
On 10 different test models, Fleat was [insert percentage accuracy]% accurate compared to a Keras Hyperband tuner, while being [insert accuracy]x faster. 
These results can be replicated in the "test.py" file.

Unlike most other tuners (such as Keras Tuner), Fleat solely supports learning rate prediction for CNNs. 
Although this cuts down on the possible use cases for Fleat, it also ensures that Fleat is exceptionally fast at it's task. 
Fleat is intended to be used alongside other tuners, with Fleat's purpose being to help shave a few minutes or hours off of the computational time.

How it works
============

Fleat uses a pretrained predictor model that takes in **100 sample images, the type of optimizer, and the loss**, as input. 
The sample images do not need to be preprocessed, Fleat automatically converts any image set following the shape **(a, x, y, c)** (with a being the number of images, 
x and y being the dimensions, and c being the color channels) to shape **(100, 224, 224, 1)**. The predictor model outputs a predicted best learning rate.

Install Fleat
=============

Fleat works with the usual pip install:

.. code-block:: console

    pip install fleat

Fleat Syntax
============

The following code snippet shows how Fleat can be integrated into the traditional MNIST solution.

..  code-block:: python
    :emphasize-lines: 1, 9

    import fleat
    import tensorflow as tf
    (train_imgs, _), (_, _) = tf.keras.datasets.fashion_mnist.load_data()
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])
    learning_rate = fleat.getLR(train_imgs, model, "adam")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )
    # ...Rest of preprocessing and training...

Applications
============

Fleat (currently) works only for simple CNNs using default Keras optimizers. The following optimizers are supported by Fleat:

* SGD
* RMSprop
* Adam
* Adadelta
* Adagrad
* Adamax
* Nadam
* Ftrl

Dependencies
============

* TensorFlow >= 2.2.0
* Keras >= 2.4.0
* NumPy >= 1.19.0
* OpenCV Python >= 4.3.0

GitHub Repo
============

The source code for the Fleat library itself can be found here: https://github.com/Hudson-Liu/fleat

Since the library itself is just a model loader, the code used to train the model and all additional required files can be found here: https://github.com/Hudson-Liu/Fleat-Model-Trainer