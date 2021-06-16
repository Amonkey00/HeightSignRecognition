
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint,TensorBoard
import numpy as np
import models
import dataset
from absl import app, flags, logging
from absl.flags import FLAGS
from matplotlib import pyplot as plt


def saveFigure(history,figName='A&L_figure'):
    # show the figure
    loss = history.history['loss'][1:]
    plt.plot(loss, label='Training Loss')
    if 'val_loss' in history.history:
        val_loss = history.history['val_loss'][1:]
        plt.plot(val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('../figure/{}.jpg'.format(figName))


flags.DEFINE_float('learning_rate',1e-3,'The learning rate of model')
flags.DEFINE_integer('buffer_size',800,'The buffer_size')
flags.DEFINE_integer('batch_size',16,'The batch_size')
flags.DEFINE_integer('epochs',20,'The number of epoch')
flags.DEFINE_string('checkpoint_save_path','../checkpoints/PatchNet_v1.tf','checkpoint save path')

def main(_argv):

    # Get DataSet
    data_dir = data_dir = 'F:/HeightLimit'
    train_dataset = dataset.getTrainDataset(data_dir)
    train_dataset = train_dataset.shuffle(buffer_size=FLAGS.buffer_size,
                                          reshuffle_each_iteration=True)
    train_dataset = train_dataset.batch(FLAGS.batch_size)
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # Start to train model
    model = models.PatchNet(num_classes=2)
    optimizer = tf.keras.optimizers.Adam(lr=FLAGS.learning_rate)
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False))
    callbacks = [
            ModelCheckpoint(filepath=FLAGS.checkpoint_save_path,
                            save_best_only=True, save_weights_only=True),
            # TensorBoard(log_dr='logs')
        ]

    history = model.fit(train_dataset,epochs=FLAGS.epochs,callbacks=callbacks)
    model.summary()
    saveFigure(history,"PatchNet")


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
