import sys
import pickle
import json
import traceback

# from pathlib import Path

import numpy as np
import pandas as pd
from hyperopt import STATUS_OK, Trials, tpe, fmin
from timeit import default_timer as timer
from tools import hyper_spaces as model_space, transcript
from tools.utils import (
    load_flatten_inputs,
    scale_normalize,
    evaluate_EQ_predicttion,
    evaluate_model,
    parse_args,
    files_to_save,
    send_err_email,
    send_finish_email,
)


# sys.path.append(Path(__file__).parent.parent)
# sys.path.insert(0, Path(__file__).parent.parent)


def objective(params):
    """Objective function for Hyperparameter Optimization"""

    global ITERATION, best_metric_, shape_train_X, shape_test_X

    start = timer()
    # global best_metric_
    ITERATION += 1

    print(params)

    # store params
    df_params = pd.DataFrame([json.dumps(params)], columns=["params"])

    # scale_normalize
    train_X_, test_X_ = scale_normalize(params, train_X, test_X)

    # build and train model
    if model_name_ in ml_models:
        yhat_pred = clfs[model_name_](
            model_name_, params, train_X_, test_X_, train_y
        )
    elif model_name_ == "DNN":
        yhat_pred, current_model = clfs[model_name_](
            params, train_X_, test_X_, train_y
        )
    else:
        yhat_pred, current_model = clfs[model_name_](
            params, train_X_, test_X_, shape_train_X, shape_test_X, train_y
        )

    yhat_classes = yhat_pred[:, 3]

    yhat_pred = np.append(yhat_pred, test_y[:, None], axis=1)
    yhat_pred_all.append(yhat_pred)

    # evaluate model
    data_scores, F1_score = evaluate_model(test_y, yhat_classes)
    EQ_results, EQ_mcc = evaluate_EQ_predicttion(
        test_y, test_EQ_LABELS, test_EQ_NUM, yhat_classes
    )
    run_time = timer() - start

    # Loss must be minimized
    loss = -F1_score

    # result to save
    df_info = pd.DataFrame(
        [[ITERATION, loss, run_time]], columns=["iteration", "loss", "train_time"]
    )
    result_to_save = pd.concat([df_info, df_params, data_scores, EQ_results], axis=1)

    # change order, papered all saved file
    order = [
        "iteration",
        "loss",
        "train_time",
        "params",
        "MCC",
        "F1 score",
        "Balanced Accuracy",
        "Accuracy",
        "Precision",
        "Sensitivity",
        "EQ_MCC",
        "EQ_F1",
        "EQ_Accuracy",
        "EQ_Precision",
        "EQ_Sensitivity",
        "EQ_PR_AUC",
        "EQ_ROC_AUC",
        "EQ_true",
        "EQ_pred_proba",
        "EQ_pred_classes",
        "test_y_true",
        "test_y_pred",
    ]
    result_to_save = result_to_save[order]

    # papered metrics save file
    metrics_order = [
        "iteration",
        "loss",
        "train_time",
        "params",
        "MCC",
        "F1 score",
        "Balanced Accuracy",
        "Accuracy",
        "Precision",
        "Sensitivity",
        "EQ_MCC",
        "EQ_F1",
        "EQ_Accuracy",
        "EQ_Precision",
        "EQ_Sensitivity",
        "EQ_PR_AUC",
        "EQ_ROC_AUC",
    ]
    metrics_to_save = result_to_save[metrics_order]

    # save result files
    if ITERATION == 1:
        result_to_save.to_csv(out_file_csv, index=False, header=True)
        metrics_to_save.to_csv(metrics_csv, index=False, header=True)
        if model_name_ not in ml_models:
            best_metric_ = F1_score
            current_model.save(model_save_file)
    else:
        result_to_save.to_csv(out_file_csv, index=False, mode="a", header=False)
        metrics_to_save.to_csv(metrics_csv, index=False, mode="a", header=False)
        if model_name_ not in ml_models:
            if F1_score > best_metric_:
                current_model.save(model_save_file)
                best_metric_ = F1_score

    # Dictionary with information for evaluation
    return {
        "loss": loss,
        "params": params,
        "iteration": ITERATION,
        "train_time": run_time,
        "status": STATUS_OK,
    }


def run_a_trial():
    """Run one TPE meta optimisation step and save its results."""
    global ITERATION, ml_models, clfs, params
    ITERATION = 0
    MAX_EVALS = 100

    # optimization algorithm
    tpe_algorithm = tpe.suggest

    # Keep track of results
    bayes_trials = Trials()

    # Run optimization
    best = fmin(
        fn=objective,
        space=params[model_name_],
        algo=tpe_algorithm,
        trials=bayes_trials,
        max_evals=MAX_EVALS,
        rstate=np.random.RandomState(50),
    )

    # save the trials object
    with open(out_file_pkl, "wb") as f:
        pickle.dump(yhat_pred_all, f)

    # # load additional module
    # import pickle
    #
    # with open('listfile.data', 'rb') as filehandle:
    #     # read the data as binary data stream
    #     placesList = pickle.load(filehandle)

    # send finish email
    print("\nOPTIMIZATION STEP COMPLETE.\n")
    send_finish_email(name=model_name_ + "_" + train_dataset_ + "_" + test_dataset_)
    transcript.stop()


if __name__ == "__main__":
    """Plot the model and run the optimisation forever (and saves results)."""

    try:
        global model_name_, train_dataset_, test_dataset_, shape_train_X, shape_test_X, train_y, test_y, log_file, out_file_csv, metrics_csv, out_file_pkl, model_save_file

        args = parse_args()
        train_dataset_ = args.train_dataset
        test_dataset_ = args.test_dataset
        model_name_ = args.model_name

        # FileNames to save results
        (
            log_file,
            out_file_csv,
            metrics_csv,
            out_file_pkl,
            model_save_file,
        ) = files_to_save(model_name_, train_dataset_, test_dataset_)

        # Save log file
        transcript.start(log_file)

        print("train dataset :", train_dataset_, end="\n")
        print("for_test dataset :", test_dataset_, end="\n")
        # print("model name :", model_name_, end="\n")

        # Optimize a new model with the TPE Algorithm:
        print("\nOPTIMIZING MODEL:", model_name_, end="\n")

        # Initialization MLmodels_name, models, clfs, params
        ml_models, _, clfs, params = model_space.initialize_models_sapces()

        # load and flatten inputs
        train_X, shape_train_X, train_y = load_flatten_inputs(train_dataset_, "train")
        test_X, shape_test_X, test_y, test_EQ_LABELS, test_EQ_NUM = load_flatten_inputs(
            test_dataset_, "for_test"
        )

        yhat_pred_all = []

        # run trials
        run_a_trial()

    except Exception as err:
        send_err_email(
            name=model_name_ + "_" + train_dataset_ + "_" + test_dataset_,
            ex=repr(err),
            ex_detail=traceback.format_exc(),
        )
        print(traceback.format_exc(), end="\n")
        err_str = str(err)
        print(err_str, end="\n")
        traceback_str = str(traceback.format_exc())
        print(traceback_str, end="\n")
