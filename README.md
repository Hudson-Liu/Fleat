# Fleat 
Fleat: **F**ast **Lea**rning Rate **T**uner

# Motivation
Traditional hyperparameter tuners suffer from their long runtimes and high resource consumption; even the best tuners (such as the Hyperband Tuner) require training the model 2 or 3 times over. Although not as generalizable as hyperparameter tuners, Fleat attempts to make learning rate optimization much faster and less resource heavy. On 10 different test models, Fleat was [insert percentage accuracy]% accurate compared to a Keras Hyperband tuner, while being [insert accuracy]x faster. These results can be replicated in the "test.py" file.

# Install Fleat
```pip install fleat```

# Fleat Syntax
The following code shows how Fleat can be integrated into the traditional MNIST solution.
    import fleat
    import tensorflow as tf
    (train_imgs, \_), (\_,\_) = tf.keras.datasets.fashion_mnist.load_data()
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])
    learning_rate = fleat.getLR(model, train_imgs, "adam")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )
    # ...Rest of preprocessing and training...

# Applications
Fleat (currently) works only for simple CNNs using default Keras optimizers. The following optimizers are supported by Fleat:
- SGD
- RMSprop
- Adam
- Adadelta
- Adagrad
- Adamax
- Nadam
- Ftrl

# How it works
Fleat uses a pretrained predictor model that takes in **100 sample images, the type of optimizer, and the loss**, as input. The sample images do not need to be preprocessed, Fleat automatically converts any image set following the shape **(a, x, y, c)** (with a being the number of images, x and y being the dimensions, and c being the color channels) to shape **(100, 224, 224, 1)**. The predictor model outputs a predicted best learning rate.
