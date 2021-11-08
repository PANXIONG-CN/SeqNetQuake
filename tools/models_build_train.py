from keras.layers import (
    LSTM,
    Conv1D,
    MaxPooling1D,
    TimeDistributed,
    ConvLSTM2D,
    Bidirectional,
)
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD, Adadelta, Adam  # , rmsprop
from keras.layers.normalization import BatchNormalization
from keras.utils import to_categorical
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import numpy as np


def ml_build_and_train(model_, params, train_X_, test_X_, train_y):
    if model_ == "RF":
        model = RandomForestClassifier(**params)
    elif model_ == "LR":
        model = LogisticRegression(**params, solver="lbfgs", multi_class="auto")
    elif model_ == "NB":
        model = BernoulliNB(**params)
    elif model_ == "SVM":
        model = SVC(**params)
    elif model_ == "KNN":
        model = KNeighborsClassifier(**params)
    elif model_ == "DTREE":
        model = DecisionTreeClassifier(**params)
    elif model_ == "GBM":
        model = GradientBoostingClassifier(**params)

    # train_y = to_categorical(train_y)
    model.fit(train_X_, train_y)
    yhat_proba = model.predict_proba(test_X_)
    yhat_classes = np.argmax(yhat_proba, axis=1)

    # yhat_classes = model.predict(test_X_)

    yhat_pred = np.append(yhat_proba, yhat_classes[:, None], axis=1)

    return yhat_pred


def dnn_build_and_train(params, train_X_, test_X_, train_y):
    # zero-offset class values
    # train_y = train_y - 1
    train_y = to_categorical(train_y)

    n_outputs = train_y.shape[1]

    model = Sequential()
    model.add(
        Dense(
            units=params["units1"],
            input_dim=train_X_.shape[1],
            kernel_initializer="glorot_uniform",
        )
    )
    model.add(Activation(params["activation"]))
    model.add(Dropout(params["dropout1"]))
    model.add(BatchNormalization())

    model.add(Dense(units=params["units2"], kernel_initializer="glorot_uniform"))
    model.add(Activation(params["activation"]))
    model.add(Dropout(params["dropout2"]))
    model.add(BatchNormalization())

    if params["choice"]["layers"] == "three":
        model.add(
            Dense(
                units=params["choice"]["units3_3"], kernel_initializer="glorot_uniform",
            )
        )
        model.add(Activation(params["activation"]))
        model.add(Dropout(params["choice"]["dropout3_3"]))
        model.add(BatchNormalization())
    elif params["choice"]["layers"] == "four":
        model.add(
            Dense(
                units=params["choice"]["units3_4"], kernel_initializer="glorot_uniform",
            )
        )
        model.add(Activation(params["activation"]))
        model.add(Dropout(params["choice"]["dropout3_4"]))
        model.add(BatchNormalization())

        model.add(
            Dense(
                units=params["choice"]["units4_4"], kernel_initializer="glorot_uniform",
            )
        )
        model.add(Activation(params["activation"]))
        model.add(Dropout(params["choice"]["dropout4_4"]))
        model.add(BatchNormalization())
    elif params["choice"]["layers"] == "five":
        model.add(
            Dense(
                units=params["choice"]["units3_5"], kernel_initializer="glorot_uniform",
            )
        )
        model.add(Activation(params["activation"]))
        model.add(Dropout(params["choice"]["dropout3_5"]))
        model.add(BatchNormalization())

        model.add(
            Dense(
                units=params["choice"]["units4_5"], kernel_initializer="glorot_uniform",
            )
        )
        model.add(Activation(params["activation"]))
        model.add(Dropout(params["choice"]["dropout4_5"]))
        model.add(BatchNormalization())

        model.add(
            Dense(
                units=params["choice"]["units5_5"], kernel_initializer="glorot_uniform",
            )
        )
        model.add(Activation(params["activation"]))
        model.add(Dropout(params["choice"]["dropout5_5"]))
        model.add(BatchNormalization())
    elif params["choice"]["layers"] == "six":
        model.add(
            Dense(
                units=params["choice"]["units3_6"], kernel_initializer="glorot_uniform",
            )
        )
        model.add(Activation(params["activation"]))
        model.add(Dropout(params["choice"]["dropout3_6"]))
        model.add(BatchNormalization())

        model.add(
            Dense(
                units=params["choice"]["units4_6"], kernel_initializer="glorot_uniform",
            )
        )
        model.add(Activation(params["activation"]))
        model.add(Dropout(params["choice"]["dropout4_6"]))
        model.add(BatchNormalization())

        model.add(
            Dense(
                units=params["choice"]["units5_6"], kernel_initializer="glorot_uniform",
            )
        )
        model.add(Activation(params["activation"]))
        model.add(Dropout(params["choice"]["dropout5_6"]))
        model.add(BatchNormalization())

        model.add(
            Dense(
                units=params["choice"]["units6_6"], kernel_initializer="glorot_uniform",
            )
        )
        model.add(Activation(params["activation"]))
        model.add(Dropout(params["choice"]["dropout6_6"]))
        model.add(BatchNormalization())

    model.add(
        Dense(n_outputs, kernel_initializer="glorot_uniform", activation="softmax")
    )

    # Seleziona l'ottimizzatore corretto
    if params["optimizer"]["name"] == "adam":
        opt = Adam(lr=params["optimizer"]["learning_rate"])
    else:
        opt = SGD(
            lr=params["optimizer"]["learning_rate"],
            momentum=params["optimizer"]["momentum"],
        )

    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    model.fit(
        train_X_,
        train_y,
        epochs=params["nb_epochs"],
        batch_size=params["batch_size"],
        verbose=0,
    )

    # validation_data=(testX, testY) https://blog.csdn.net/qq_27871973/article/details/84955977

    # yhat_classes_1 = model.predict_classes(
    #     test_X_, batch_size=params["batch_size"], verbose=0
    # )

    yhat_proba = model.predict(
        test_X_, batch_size=params["batch_size"], verbose=0
    )
    yhat_classes = np.argmax(yhat_proba, axis=1)
    yhat_pred = np.append(yhat_proba, yhat_classes[:, None], axis=1)

    # yhat_classes = yhat_classes + 1

    return yhat_pred, model


def lstm_build_and_train(
        params, train_X_, test_X_, shape_train_X, shape_test_X, train_y
):
    # reshape train_X_, test_X_
    train_X_ = train_X_.reshape(shape_train_X)
    test_X_ = test_X_.reshape(shape_test_X)

    # zero-offset class values
    # train_y = train_y - 1
    # one hot encode y
    train_y = to_categorical(train_y)

    n_timesteps, n_features, n_outputs = (
        train_X_.shape[1],
        train_X_.shape[2],
        train_y.shape[1],
    )

    model = Sequential()
    if params["num_lstm_layers"] == 1:
        model.add(
            LSTM(params["units_lstm_layers"], input_shape=(n_timesteps, n_features))
        )
    else:
        model.add(
            LSTM(
                params["units_lstm_layers"],
                return_sequences=True,
                input_shape=(n_timesteps, n_features),
            )
        )
        for _ in range(2, params["num_lstm_layers"]):
            model.add(LSTM(params["units_lstm_layers"], return_sequences=True))
        model.add(LSTM(params["units_lstm_layers"]))

    model.add(Dropout(params["dropout"]))

    for _ in range(1, params["num_fc_layers"]):
        model.add(Dense(params["units_fc_layers"], activation=params["activation"]))
    model.add(Dense(n_outputs, activation="softmax"))

    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    model.fit(
        train_X_,
        train_y,
        epochs=params["nb_epochs"],
        batch_size=params["batch_size"],
        verbose=0,
    )

    # validation_data=(testX, testY) https://blog.csdn.net/qq_27871973/article/details/84955977

    # yhat_classes = model.predict_classes(
    #     test_X_, batch_size=params["batch_size"], verbose=0
    # )

    yhat_proba = model.predict(
        test_X_, batch_size=params["batch_size"], verbose=0
    )
    yhat_classes = np.argmax(yhat_proba, axis=1)
    yhat_pred = np.append(yhat_proba, yhat_classes[:, None], axis=1)

    # yhat_classes = yhat_classes + 1

    return yhat_pred, model

    # yhat_classes = yhat_classes + 1

    # return yhat_classes, model


def cnn_build_and_train(
        params, train_X_, test_X_, shape_train_X, shape_test_X, train_y
):
    # reshape train_X_, test_X_
    train_X_ = train_X_.reshape(shape_train_X)
    test_X_ = test_X_.reshape(shape_test_X)

    # zero-offset class values
    # train_y = train_y - 1
    # one hot encode y
    train_y = to_categorical(train_y)

    n_timesteps, n_features, n_outputs = (
        train_X_.shape[1],
        train_X_.shape[2],
        train_y.shape[1],
    )
    model = Sequential()

    model.add(
        Conv1D(
            filters=params["num_filters_cnn"],
            kernel_size=params["size_kernel_cnn"],
            activation=params["activation_cnn"],
            input_shape=(n_timesteps, n_features),
        )
    )
    for _ in range(1, params["num_cnn_layers"]):
        model.add(
            Conv1D(
                filters=params["num_filters_cnn"],
                kernel_size=params["size_kernel_cnn"],
                activation=params["activation_cnn"],
            )
        )

    model.add(Dropout(params["dropout"]))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())

    for _ in range(1, params["num_fc_layers"]):
        model.add(Dense(params["units_fc_layers"], activation=params["activation_fc"]))
    model.add(Dense(n_outputs, activation="softmax"))

    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    model.fit(
        train_X_,
        train_y,
        epochs=params["nb_epochs"],
        batch_size=params["batch_size"],
        verbose=0,
    )

    yhat_proba = model.predict(
        test_X_, batch_size=params["batch_size"], verbose=0
    )
    yhat_classes = np.argmax(yhat_proba, axis=1)
    yhat_pred = np.append(yhat_proba, yhat_classes[:, None], axis=1)

    return yhat_pred, model


def cnnlstm_build_and_train(
        params, train_X_, test_X_, shape_train_X, shape_test_X, train_y
):
    # reshape train_X_, test_X_
    train_X_ = train_X_.reshape(shape_train_X)
    test_X_ = test_X_.reshape(shape_test_X)

    # zero-offset class values
    # train_y = train_y - 1
    # one hot encode y
    train_y = to_categorical(train_y)

    n_timesteps, n_features, n_outputs = (
        train_X_.shape[1],
        train_X_.shape[2],
        train_y.shape[1],
    )

    # reshape data into time steps of sub-sequences
    # n_steps = params["n_steps"]
    # if n_steps == 2:
    #     n_length = 10
    # else:
    #     n_length = 5

    n_steps = 2
    n_length = 30
    train_X_ = train_X_.reshape((train_X_.shape[0], n_steps, n_length, n_features))
    test_X_ = test_X_.reshape((test_X_.shape[0], n_steps, n_length, n_features))

    # define model
    model = Sequential()

    # model.add(
    #     TimeDistributed(Conv1D(filters=4, kernel_size=3, activation='relu'), input_shape=(None, n_length, n_features)))
    # model.add(TimeDistributed(Conv1D(filters=4, kernel_size=3, activation='relu')))
    # model.add(TimeDistributed(Conv1D(filters=4, kernel_size=3, activation='relu')))
    model.add(
        TimeDistributed(
            Conv1D(
                filters=params["num_filters_cnn"],
                kernel_size=params["size_kernel_cnn"],
                activation=params["activation_cnn"],
            ),
            input_shape=(None, n_length, n_features),
        )
    )
    for _ in range(1, params["num_cnn_layers"]):
        model.add(
            TimeDistributed(
                Conv1D(
                    filters=params["num_filters_cnn"],
                    kernel_size=params["size_kernel_cnn"],
                    activation=params["activation_cnn"],
                )
            )
        )
    model.add(TimeDistributed(Dropout(params["dropout"])))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2, padding="same")))
    model.add(TimeDistributed(Flatten()))

    for _ in range(1, params["num_lstm_layers"]):
        model.add(LSTM(params["units_lstm_layers"], return_sequences=True))
    model.add(LSTM(params["units_lstm_layers"]))

    model.add(Dropout(params["dropout"]))

    for _ in range(1, params["num_fc_layers"]):
        model.add(Dense(params["units_fc_layers"], activation=params["activation_fc"]))
    model.add(Dense(n_outputs, activation="softmax"))

    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    model.fit(
        train_X_,
        train_y,
        epochs=params["nb_epochs"],
        batch_size=params["batch_size"],
        verbose=0,
    )

    # yhat_classes = model.predict_classes(
    #     test_X_, batch_size=params["batch_size"], verbose=0
    # )

    yhat_proba = model.predict(
        test_X_, batch_size=params["batch_size"], verbose=0
    )
    yhat_classes = np.argmax(yhat_proba, axis=1)
    yhat_pred = np.append(yhat_proba, yhat_classes[:, None], axis=1)

    # yhat_classes = yhat_classes + 1

    return yhat_pred, model

    # yhat_classes = yhat_classes + 1

    # return yhat_classes, model


def convlstm_build_and_train(
        params, train_X_, test_X_, shape_train_X, shape_test_X, train_y
):
    # reshape train_X_, test_X_
    train_X_ = train_X_.reshape(shape_train_X)
    test_X_ = test_X_.reshape(shape_test_X)

    # zero-offset class values
    # train_y = train_y - 1
    # one hot encode y
    train_y = to_categorical(train_y)

    n_timesteps, n_features, n_outputs = (
        train_X_.shape[1],
        train_X_.shape[2],
        train_y.shape[1],
    )

    # reshape data into time steps of sub-sequences
    n_steps, n_length = 4, 5
    n_steps, n_length = 4, 5
    train_X_ = train_X_.reshape((train_X_.shape[0], n_steps, 1, n_length, n_features))
    test_X_ = test_X_.reshape((test_X_.shape[0], n_steps, 1, n_length, n_features))

    # define model
    model = Sequential()
    model.add(
        ConvLSTM2D(
            filters=params["num_filters_cnn"],
            kernel_size=(1, 2),
            activation=params["activation_cnn"],
            return_sequences=True,
            input_shape=(n_steps, 1, n_length, n_features),
        )
    )

    if params["num_cnn_layers"] == 1:
        model.add(
            ConvLSTM2D(
                filters=params["num_filters_cnn"],
                kernel_size=(1, 2),
                activation=params["activation_cnn"],
                # input_shape=(n_steps, 1, n_length, n_features),
            )
        )
    else:
        model.add(
            ConvLSTM2D(
                filters=params["num_filters_cnn"],
                kernel_size=(1, 2),
                activation=params["activation_cnn"],
                return_sequences=True,
                # input_shape=(n_steps, 1, n_length, n_features),
            )
        )
        for _ in range(2, params["num_cnn_layers"]):
            model.add(
                ConvLSTM2D(
                    filters=params["num_filters_cnn"],
                    kernel_size=(1, 2),
                    activation=params["activation_cnn"],
                    return_sequences=True,
                )
            )
        model.add(
            ConvLSTM2D(
                filters=params["num_filters_cnn"],
                kernel_size=(1, 2),
                activation=params["activation_cnn"],
            )
        )

        # model.add(LSTM(params["units_lstm_layers"]))

    model.add(Dropout(params["dropout"]))
    model.add(Flatten())
    for _ in range(1, params["num_fc_layers"]):
        model.add(Dense(params["units_fc_layers"], activation=params["activation_fc"]))
    model.add(Dense(n_outputs, activation="softmax"))

    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    model.fit(
        train_X_,
        train_y,
        epochs=params["nb_epochs"],
        batch_size=params["batch_size"],
        verbose=0,
    )

    # yhat_classes = model.predict_classes(
    #     test_X_, batch_size=params["batch_size"], verbose=0
    # )

    yhat_proba = model.predict(
        test_X_, batch_size=params["batch_size"], verbose=0
    )
    yhat_classes = np.argmax(yhat_proba, axis=1)
    yhat_pred = np.append(yhat_proba, yhat_classes[:, None], axis=1)

    # yhat_classes = yhat_classes + 1

    return yhat_pred, model

    # yhat_classes = yhat_classes + 1
    #
    # return yhat_classes, model


def bilstm_build_and_train(
        params, train_X_, test_X_, shape_train_X, shape_test_X, train_y
):
    # reshape train_X_, test_X_
    train_X_ = train_X_.reshape(shape_train_X)
    test_X_ = test_X_.reshape(shape_test_X)

    # zero-offset class values
    # train_y = train_y - 1
    # one hot encode y
    train_y = to_categorical(train_y)

    n_timesteps, n_features, n_outputs = (
        train_X_.shape[1],
        train_X_.shape[2],
        train_y.shape[1],
    )

    model = Sequential()
    if params["num_lstm_layers"] == 1:
        model.add(
            Bidirectional(
                LSTM(params["units_lstm_layers"], input_shape=(n_timesteps, n_features))
            )
        )
    else:
        model.add(
            Bidirectional(
                LSTM(
                    params["units_lstm_layers"],
                    return_sequences=True,
                    input_shape=(n_timesteps, n_features),
                )
            )
        )
        for _ in range(2, params["num_lstm_layers"]):
            model.add(
                Bidirectional(LSTM(params["units_lstm_layers"], return_sequences=True))
            )
        model.add(Bidirectional(LSTM(params["units_lstm_layers"])))

    model.add(Dropout(params["dropout"]))

    for _ in range(1, params["num_fc_layers"]):
        model.add(Dense(params["units_fc_layers"], activation=params["activation_fc"]))
    model.add(Dense(n_outputs, activation="softmax"))

    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    model.fit(
        train_X_,
        train_y,
        epochs=params["nb_epochs"],
        batch_size=params["batch_size"],
        verbose=0,
    )

    # validation_data=(testX, testY) https://blog.csdn.net/qq_27871973/article/details/84955977

    # yhat_classes = model.predict_classes(
    #     test_X_, batch_size=params["batch_size"], verbose=0
    # )

    yhat_proba = model.predict(
        test_X_, batch_size=params["batch_size"], verbose=0
    )
    yhat_classes = np.argmax(yhat_proba, axis=1)
    yhat_pred = np.append(yhat_proba, yhat_classes[:, None], axis=1)

    # yhat_classes = yhat_classes + 1

    return yhat_pred, model

    # yhat_classes = yhat_classes + 1
    #
    # return yhat_classes, model


def cnnbilstm_build_and_train(
        params, train_X_, test_X_, shape_train_X, shape_test_X, train_y
):
    # reshape train_X_, test_X_
    train_X_ = train_X_.reshape(shape_train_X)
    test_X_ = test_X_.reshape(shape_test_X)

    # zero-offset class values
    # train_y = train_y - 1
    # one hot encode y
    train_y = to_categorical(train_y)

    n_timesteps, n_features, n_outputs = (
        train_X_.shape[1],
        train_X_.shape[2],
        train_y.shape[1],
    )

    # reshape data into time steps of sub-sequences
    # n_steps = params["n_steps"]
    # if n_steps == 2:
    #     n_length = 10
    # else:
    #     n_length = 5
    n_steps = 2
    n_length = 10
    train_X_ = train_X_.reshape((train_X_.shape[0], n_steps, n_length, n_features))
    test_X_ = test_X_.reshape((test_X_.shape[0], n_steps, n_length, n_features))

    # define model
    model = Sequential()
    model.add(
        TimeDistributed(
            Conv1D(
                filters=params["num_filters_cnn"],
                kernel_size=params["size_kernel_cnn"],
                activation=params["activation_cnn"],
            ),
            input_shape=(None, n_length, n_features),
        )
    )
    for _ in range(1, params["num_cnn_layers"]):
        model.add(
            TimeDistributed(
                Conv1D(
                    filters=params["num_filters_cnn"],
                    kernel_size=params["size_kernel_cnn"],
                    activation=params["activation_cnn"],
                )
            )
        )
    model.add(TimeDistributed(Dropout(params["dropout"])))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten()))

    for _ in range(1, params["num_lstm_layers"]):
        model.add(
            Bidirectional(LSTM(params["units_lstm_layers"], return_sequences=True))
        )
    model.add(Bidirectional(LSTM(params["units_lstm_layers"])))

    model.add(Dropout(params["dropout"]))

    for _ in range(1, params["num_fc_layers"]):
        model.add(Dense(params["units_fc_layers"], activation=params["activation_fc"]))
    model.add(Dense(n_outputs, activation="softmax"))

    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    model.fit(
        train_X_,
        train_y,
        epochs=params["nb_epochs"],
        batch_size=params["batch_size"],
        verbose=0,
    )

    # yhat_classes_1 = model.predict_classes(
    #     test_X_, batch_size=params["batch_size"], verbose=0
    # )

    yhat_proba = model.predict(
        test_X_, batch_size=params["batch_size"], verbose=0
    )
    yhat_classes = np.argmax(yhat_proba, axis=1)
    yhat_pred = np.append(yhat_proba, yhat_classes[:, None], axis=1)

    # yhat_classes = yhat_classes + 1

    return yhat_pred, model
