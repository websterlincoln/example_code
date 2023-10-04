import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import Precision, Recall, AUC
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler
from tensorflow import summary

def parse_function(example_proto):
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    features = tf.io.parse_single_example(example_proto, feature_description)
    image = tf.io.decode_jpeg(features['image'], channels=3)
    image = tf.image.resize(image, [256, 256])
    label = features['label']
    return image, label

def load_tfrecords_dataset(file_pattern, batch_size, buffer_size, num_epochs=None):
    dataset = tf.data.TFRecordDataset(file_pattern)
    dataset = dataset.map(parse_function)
    dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

batch_size_per_gpu = 32
num_gpus = len(tf.config.list_physical_devices('GPU'))
total_batch_size = batch_size_per_gpu * num_gpus

output_data_dir = '/tf/notebooks/test-image/CAMELYON16/tf_records'

shuffle_buffer_size = 1000

train_dataset = load_tfrecords_dataset(os.path.join(output_data_dir, 'train.tfrecords'), total_batch_size, shuffle_buffer_size)
val_dataset = load_tfrecords_dataset(os.path.join(output_data_dir, 'val.tfrecords'), total_batch_size, shuffle_buffer_size)
test_dataset = load_tfrecords_dataset(os.path.join(output_data_dir, 'test.tfrecords'), total_batch_size, shuffle_buffer_size)

def load_num_samples_from_json(file_path):
    with open(file_path, 'r') as f:
        num_samples = json.load(f)
    return num_samples

num_samples_file = os.path.join(output_data_dir, 'num_samples.json')
num_samples = load_num_samples_from_json(num_samples_file)

num_train_samples = num_samples['train']
num_val_samples = num_samples['val']
num_test_samples = num_samples['test']

train_samples = np.array([(x, y) for x, y in train_dataset.as_numpy_iterator()])
class_counts = np.bincount(train_samples[:, 1], minlength=2)
total_samples = train_samples.shape[0]
class_weights = total_samples / (2.0 * class_counts)


strategy = tf.distribute.MirroredStrategy(devices=["GPU:0", "GPU:1", "GPU:2"])

with strategy.scope():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x) # Add dropout layer to prevent overfitting
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    precision = Precision(name='precision')
    recall = Recall(name='recall')
    auc = AUC(name='auc')

    optimizer = Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy', precision, recall, auc]
                  )
def step_decay(epoch):
    initial_lr = 0.0005
    drop_factor = 0.5
    epochs_drop = 4
    lr = initial_lr * np.power(drop_factor, np.floor((1 + epoch) / epochs_drop))

    with summary.create_file_writer(log_dir).as_default():
        summary.scalar('learning rate', data=lr, step=epoch)

    return lr

lr_scheduler = LearningRateScheduler(step_decay, verbose=1)

log_dir = '/tf/notebooks/data-science/notebooks/tensorflow/logs'
tensorboard_callback = TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,
    update_freq='epoch',
    write_images=True,
    profile_batch=0
)

checkpoint_callback = ModelCheckpoint(
    filepath='/tf/notebooks/data-science/notebooks/tensorflow/model/weights.{epoch:02d}-{val_loss:.2f}.h5',
    save_weights_only=True,
    save_best_only=True,
    monitor='val_loss',
    mode='min'
)



history = model.fit(
    train_dataset,
    epochs=20,
    steps_per_epoch=num_train_samples // total_batch_size,
    validation_data=val_dataset,
    validation_steps=num_val_samples // total_batch_size,
    callbacks=[tensorboard_callback, checkpoint_callback, lr_scheduler],
    class_weight=class_weights
)

test_loss, test_acc, test_precision, test_recall, test_auc = model.evaluate(test_dataset)
print(f'Test accuracy: {test_acc}, Test loss: {test_loss}, Test precision: {test_precision}, Test recall: {test_recall}, Test AUC: {test_auc}')

model.save('/tf/notebooks/data-science/notebooks/tensorflow/saved_model/my_model')

print('Done')
