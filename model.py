from parameters import *
from tools.data_preprocessor import DataPreprocessor
from tools.data_provider import DataProvider
from tools.network import Network
from keras.models import load_model
from tools.train_history import TrainHistory

network = Network()

model = network.create_convolutional_nvidia_style_with_regularization_cropping(input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3), num_classes=NUM_CLASSES)
#model = network.simple(input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3), num_classes=NUM_CLASSES)
#model = load_model('./models/T6/model-0063-0.0731.h5')
# model = load_model('./models/M2/model0003-2.7265-0.1635.h5')
# model = load_model('./models/M3/model0003-2.6417-0.1656.h5')
# model = load_model('./models/M4/model0006-2.7040-0.1295.h5')
# model = load_model('./models/M4B/model0010-3.0288-0.0933.h5')
# model = load_model('./models/X1/model0000-3.0109-0.2700.h5')

model.summary()

model.compile(loss='mse',
              optimizer='adam')

provider = DataProvider(DATA, LOG_FILE, TRAIN_BATCH_SIZE)
preprocessor = DataPreprocessor()
history = TrainHistory()

X_test_raw, y_test_raw = provider.get_raw_test_data()
X_test, y_test = preprocessor.preprocess(X_test_raw, y_test_raw)

# y_test = preprocessor.label_to_one_hot_encoding(y_test)

history_objects = []
test_loss_history = []

for X_train_raw, y_train_raw, count in provider.get_next_batch_of_raw_train_data():
    X_train, y_train = preprocessor.preprocess(X_train_raw, y_train_raw)

    # y_train = preprocessor.label_to_one_hot_encoding(y_train)

    history_object = model.fit(X_train, y_train,
                                 batch_size=TRAIN_KERAS_BATCH_SIZE,
                                 nb_epoch=TRAIN_KERAS_EPOCH,
                                 validation_split=0.2,
                                 shuffle=True, verbose=2)



    print('\nEvaluation on %d test samples' % X_test.shape[0])
    test_loss = model.evaluate(X_test, y_test)

    test_loss_history.append(test_loss)

    print(' - %s: %.4f' % ("loss", test_loss), end="")

    if count % 3 == 0:
        model.save('./models/model-%.4d-%.4f.h5' % (count, test_loss))

        history.save_history(history_objects, history_object, test_loss_history, label="%.4d-%.4f" % (count, test_loss))


