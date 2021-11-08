from hyperopt import hp
import numpy as np
from tools.models_build_train import (
    ml_build_and_train,
    dnn_build_and_train,
    lstm_build_and_train,
    cnn_build_and_train,
    cnnlstm_build_and_train,
    convlstm_build_and_train,
    bilstm_build_and_train,
    cnnbilstm_build_and_train,
)


def initialize_models_sapces():

    ml_models_name = ["RF", "LR", "NB", "SVM", "KNN", "DTREE", "GBM"]

    models = [
        "RF",
        "LR",
        "NB",
        "SVM",
        "KNN",
        "DTREE",
        "GBM",
        "DNN",
        "LSTM",
        "CNN",
        "CNNLSTM",
        "ConvLSTM",
        "BiLSTM",
        "CNNBiLSTM",
    ]

    clfs = {
        models[0]: ml_build_and_train,
        models[1]: ml_build_and_train,
        models[2]: ml_build_and_train,
        models[3]: ml_build_and_train,
        models[4]: ml_build_and_train,
        models[5]: ml_build_and_train,
        models[6]: ml_build_and_train,
        models[7]: dnn_build_and_train,
        models[8]: lstm_build_and_train,
        models[9]: cnn_build_and_train,
        models[10]: cnnlstm_build_and_train,
        models[11]: convlstm_build_and_train,
        models[12]: bilstm_build_and_train,
        models[13]: cnnbilstm_build_and_train,
    }

    params = {
        models[0]: {
            "max_depth": hp.choice("max_depth", range(1, 28, 1)),
            "max_features": hp.choice("max_features", range(1, 10)),
            "n_estimators": hp.choice("n_estimators", range(2, 300, 2)),
            "min_samples_split": hp.choice("min_samples_split", range(2, 150, 1)),
            "min_samples_leaf": hp.choice("min_samples_leaf", range(1, 60, 1)),
            "max_leaf_nodes": hp.choice("max_leaf_nodes", range(2, 60, 1)),
            "min_weight_fraction_leaf": hp.choice(
                "min_weight_fraction_leaf", [0.1, 0.2, 0.3, 0.4]
            ),
            "criterion": hp.choice("criterion", ["gini", "entropy"]),
            "bootstrap": hp.choice("bootstrap", [True, False]),
            "scale": hp.choice("scale", [0, 1]),
            "normalize": hp.choice("normalize", [0, 1]),
        },
        models[1]: {
            "C": hp.loguniform("C", low=-4 * np.log(10), high=4 * np.log(10)),
            "class_weight": hp.choice("class_weight", [None, "balanced"]),
            "scale": hp.choice("scale", [0, 1]),
            "normalize": hp.choice("normalize", [0, 1]),
        },
        models[2]: {
            "alpha": hp.uniform("alpha", 0.0, 2.0),
            "scale": hp.choice("scale", [0, 1]),
            "normalize": hp.choice("normalize", [0, 1]),
        },
        models[3]: {
            "C": hp.uniform("C", 0, 10.0),
            "kernel": hp.choice("kernel", ["linear", "rbf"]),
            "gamma": hp.uniform("gamma", 0, 20.0),
            "scale": hp.choice("scale", [0, 1]),
            "normalize": hp.choice("normalize", [0, 1]),
        },
        models[4]: {
            "n_neighbors": hp.choice("knn_n_neighbors", range(1, 50)),
            "scale": hp.choice("scale", [0, 1]),
            "normalize": hp.choice("normalize", [0, 1]),
        },
        models[5]: {
            "max_depth": hp.choice("max_depth", range(1, 20)),
            "max_features": hp.choice("max_features", range(1, 5)),
            "criterion": hp.choice("criterion", ["gini", "entropy"]),
            "scale": hp.choice("scale", [0, 1]),
            "normalize": hp.choice("normalize", [0, 1]),
        },
        models[6]: {
            "learning_rate": hp.uniform("lr", 0.01, 0.2),
            "subsample": hp.uniform("ss", 0.8, 1.0),
            "max_depth": hp.quniform("md", 2, 10, 1),
            "max_features": hp.choice("mf", ("sqrt", "log2", None)),
            "min_samples_leaf": hp.uniform("msl", 0.1, 0.5),
            "min_samples_split": hp.uniform("mss", 0.0, 0.5),
            "n_estimators": hp.choice("n_estimators", [10, 100, 1000]),
            "scale": hp.choice("scale", [0, 1]),
            "normalize": hp.choice("normalize", [0, 1]),
        },
        models[7]: {
            "choice": hp.choice(
                "num_layers",
                [
                    {"layers": "two",},
                    {
                        "layers": "three",
                        "units3_3": hp.choice("units3_3", [64, 128, 256, 512]),
                        "dropout3_3": hp.choice("dropout3_3", [0.25, 0.5, 0.75]),
                    },
                    {
                        "layers": "four",
                        "units3_4": hp.choice("units3_4", [64, 128, 256, 512]),
                        "dropout3_4": hp.choice("dropout3_4", [0.25, 0.5, 0.75]),
                        "units4_4": hp.choice("units4_4", [64, 128, 256, 512]),
                        "dropout4_4": hp.choice("dropout4_4", [0.25, 0.5, 0.75]),
                    },
                    {
                        "layers": "five",
                        "units3_5": hp.choice("units3_5", [64, 128, 256, 512]),
                        "dropout3_5": hp.choice("dropout3_5", [0.25, 0.5, 0.75]),
                        "units4_5": hp.choice("units4_5", [64, 128, 256, 512]),
                        "dropout4_5": hp.choice("dropout4_5", [0.25, 0.5, 0.75]),
                        "units5_5": hp.choice("units5_5", [64, 128, 256, 512]),
                        "dropout5_5": hp.choice("dropout5_5", [0.25, 0.5, 0.75]),
                    },
                    {
                        "layers": "six",
                        "units3_6": hp.choice("units3_6", [64, 128, 256, 512]),
                        "dropout3_6": hp.choice("dropout3_6", [0.25, 0.5, 0.75]),
                        "units4_6": hp.choice("units4_6", [64, 128, 256, 512]),
                        "dropout4_6": hp.choice("dropout4_6", [0.25, 0.5, 0.75]),
                        "units5_6": hp.choice("units5_6", [64, 128, 256, 512]),
                        "dropout5_6": hp.choice("dropout5_6", [0.25, 0.5, 0.75]),
                        "units6_6": hp.choice("units6_6", [64, 128, 256, 512]),
                        "dropout6_6": hp.choice("dropout6_6", [0.25, 0.5, 0.75]),
                    },
                ],
            ),
            "units1": hp.choice("units1", [64, 128, 256, 512]),
            "units2": hp.choice("units2", [64, 128, 256, 512]),
            "dropout1": hp.choice("dropout1", [0.25, 0.5, 0.75]),
            "dropout2": hp.choice("dropout2", [0.25, 0.5, 0.75]),
            "batch_size": hp.choice("batch_size", [10, 20, 40, 60, 80, 100]),
            # scope.power_of_two(hp.quniform('batch_size', 0, 8, q=1)),
            "nb_epochs": hp.choice("nb_epochs", [10, 50, 100]),
            "optimizer": opt_search_space,
            "activation": hp.choice("activation", ["relu", "tanh", "sigmoid"]),
            "scale": hp.choice("scale", [0, 1]),
            "normalize": hp.choice("normalize", [0, 1]),
        },
        models[8]: {
            "num_lstm_layers": hp.choice("num_lstm_layers", [1, 2, 3, 4, 5, 6]),
            "units_lstm_layers": hp.choice("units_lstm_layers", [64, 128, 256, 512]),
            "num_fc_layers": hp.choice("num_fc_layers", [2, 3, 4, 5, 6]),
            "units_fc_layers": hp.choice("units_fc_layers", [64, 128, 256, 512]),
            "dropout": hp.choice("dropout", [0.25, 0.5, 0.75]),
            "batch_size": hp.choice("batch_size", [10, 20, 40, 60, 80, 100]),
            "nb_epochs": hp.choice("nb_epochs", [10, 50, 100]),
            "activation": hp.choice("activation", ["relu", "tanh", "sigmoid"]),
            "scale": hp.choice("scale", [0, 1]),
            "normalize": hp.choice("normalize", [0, 1]),
        },
        models[9]: {
            "num_cnn_layers": hp.choice("num_cnn_layers", [2, 3, 4]),
            "num_filters_cnn": hp.choice("num_filters_cnn", [2, 3, 4]),
            "size_kernel_cnn": hp.choice("size_kernel_cnn", [2, 3]),
            "activation_cnn": hp.choice("activation_cnn", ["relu", "tanh", "sigmoid"]),
            "num_fc_layers": hp.choice("num_fc_layers", [2, 3, 4, 5, 6]),
            "units_fc_layers": hp.choice("units_fc_layers", [64, 128, 256, 512]),
            "activation_fc": hp.choice("activation_fc", ["relu", "tanh", "sigmoid"]),
            "dropout": hp.choice("dropout", [0.25, 0.5, 0.75]),
            "batch_size": hp.choice("batch_size", [10, 20, 40, 60, 80, 100]),
            "nb_epochs": hp.choice("nb_epochs", [10, 50, 100]),
            "scale": hp.choice("scale", [0, 1]),
            "normalize": hp.choice("normalize", [0, 1]),
        },
        models[10]: {
            "num_cnn_layers": hp.choice("num_cnn_layers", [2, 3, 4]),
            "num_filters_cnn": hp.choice("num_filters_cnn", [2, 3, 4]),
            "size_kernel_cnn": hp.choice("size_kernel_cnn", [2, 3]),
            "activation_cnn": hp.choice("activation_cnn", ["relu", "tanh", "sigmoid"]),
            "num_lstm_layers": hp.choice("num_lstm_layers", [1, 2, 3, 4, 5]),
            "units_lstm_layers": hp.choice("units_lstm_layers", [64, 128, 256, 512]),
            "num_fc_layers": hp.choice("num_fc_layers", [2, 3, 4, 5, 6]),
            "units_fc_layers": hp.choice("units_fc_layers", [64, 128, 256, 512]),
            "activation_fc": hp.choice("activation_fc", ["relu", "tanh", "sigmoid"]),
            "dropout": hp.choice("dropout", [0.25, 0.5, 0.75]),
            "batch_size": hp.choice("batch_size", [10, 20, 40, 60, 80, 100]),
            "nb_epochs": hp.choice("nb_epochs", [10, 50, 100]),
            "n_steps": hp.choice("n_steps", [2, 4]),
            "scale": hp.choice("scale", [0, 1]),
            "normalize": hp.choice("normalize", [0, 1]),
        },
        models[11]: {
            "num_cnn_layers": hp.choice("num_cnn_layers", [2, 3]),
            "num_filters_cnn": hp.choice("num_filters_cnn", [2, 3]),
            "activation_cnn": hp.choice("activation_cnn", ["relu", "tanh", "sigmoid"]),
            "num_fc_layers": hp.choice("num_fc_layers", [2, 3, 4, 5, 6]),
            "units_fc_layers": hp.choice("units_fc_layers", [64, 128, 256, 512]),
            "activation_fc": hp.choice("activation_fc", ["relu", "tanh", "sigmoid"]),
            "dropout": hp.choice("dropout", [0.25, 0.5, 0.75]),
            "batch_size": hp.choice("batch_size", [10, 20, 40, 60, 80, 100]),
            "nb_epochs": hp.choice("nb_epochs", [10, 50, 100]),
            "scale": hp.choice("scale", [0, 1]),
            "normalize": hp.choice("normalize", [0, 1]),
        },
        models[12]: {
            "num_lstm_layers": hp.choice("num_lstm_layers", [1, 2, 3, 4, 5, 6]),
            "units_lstm_layers": hp.choice("units_lstm_layers", [64, 128, 256, 512]),
            "num_fc_layers": hp.choice("num_fc_layers", [2, 3, 4, 5, 6]),
            "units_fc_layers": hp.choice("units_fc_layers", [64, 128, 256, 512]),
            "dropout": hp.choice("dropout", [0.25, 0.5, 0.75]),
            "batch_size": hp.choice("batch_size", [10, 20, 40, 60, 80, 100]),
            "nb_epochs": hp.choice("nb_epochs", [10, 50, 100]),
            "activation_fc": hp.choice("activation_fc", ["relu", "tanh", "sigmoid"]),
            "scale": hp.choice("scale", [0, 1]),
            "normalize": hp.choice("normalize", [0, 1]),
        },
        models[13]: {
            "num_cnn_layers": hp.choice("num_cnn_layers", [2, 3, 4]),
            "num_filters_cnn": hp.choice("num_filters_cnn", [2, 3, 4]),
            "size_kernel_cnn": hp.choice("size_kernel_cnn", [2, 3]),
            "activation_cnn": hp.choice("activation_cnn", ["relu", "tanh", "sigmoid"]),
            "num_lstm_layers": hp.choice("num_lstm_layers", [1, 2, 3, 4, 5]),
            "units_lstm_layers": hp.choice("units_lstm_layers", [64, 128, 256, 512]),
            "num_fc_layers": hp.choice("num_fc_layers", [2, 3, 4, 5, 6]),
            "units_fc_layers": hp.choice("units_fc_layers", [64, 128, 256, 512]),
            "activation_fc": hp.choice("activation_fc", ["relu", "tanh", "sigmoid"]),
            "dropout": hp.choice("dropout", [0.25, 0.5, 0.75]),
            "batch_size": hp.choice("batch_size", [10, 20, 40, 60, 80, 100]),
            "nb_epochs": hp.choice("nb_epochs", [10, 50, 100]),
            "n_steps": hp.choice("n_steps", [2, 4]),
            "scale": hp.choice("scale", [0, 1]),
            "normalize": hp.choice("normalize", [0, 1]),
        },
    }

    return ml_models_name, models, clfs, params


opt_search_space = hp.choice(
    "name",
    [
        {
            "name": "adam",
            "learning_rate": hp.loguniform(
                "learning_rate_adam", -10, 0
            ),  # Note the name of the label to avoid duplicates
        },
        {
            "name": "sgd",
            "learning_rate": hp.loguniform(
                "learning_rate_sgd", -15, 1
            ),  # Note the name of the label to avoid duplicates
            "momentum": hp.uniform("momentum", 0, 1.0),
        },
    ],
)
