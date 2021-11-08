import argparse
import sys
from pathlib import Path
import smtplib
from email.mime.text import MIMEText
import datetime
import traceback
import numpy as np
import pandas as pd
from sklearn.metrics import (
    matthews_corrcoef,
    f1_score,
    balanced_accuracy_score,
    accuracy_score,
    precision_score,
    recall_score,
    precision_recall_curve,
    auc,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler, Normalizer

np.seterr(divide="ignore", invalid="ignore")


def get_project_root() -> Path:
    """Returns project root folder."""
    return Path(__file__).parent.parent


def load_flatten_inputs(prefix, dataset_type):

    project_root = get_project_root()

    filesave = project_root / "input"

    # stored in npz file
    # inputs = np.load(filesave / (prefix + "_" + dataset_type + "_input.npz"))
    inputs = np.load(filesave / (prefix + "_input.npz"))

    if dataset_type == "for_test":
        test_X = inputs["test_X"]
        test_y = inputs["test_y"]
        test_EQ_LABELS = inputs["test_EQ_LABELS"]
        test_EQ_NUM = inputs["test_EQ_NUM"]

        # flatten test_X
        shape_test_X = test_X.shape
        test_X = test_X.reshape((test_X.shape[0], test_X.shape[1] * test_X.shape[2]))
        test_y = test_y[:, 0]
        test_EQ_LABELS = test_EQ_LABELS[:, 0]
        test_EQ_NUM = test_EQ_NUM[:, 0]

        return test_X, shape_test_X, test_y, test_EQ_LABELS, test_EQ_NUM
    else:
        train_X = inputs["train_X"]
        train_y = inputs["train_y"]

        # flatten train_X
        shape_train_X = train_X.shape
        train_X = train_X.reshape(
            (train_X.shape[0], train_X.shape[1] * train_X.shape[2])
        )
        train_y = train_y[:, 0]
        return train_X, shape_train_X, train_y


def scale_normalize(params, train_X, test_X):
    train_X = np.nan_to_num(train_X)
    test_X = np.nan_to_num(test_X)

    if "normalize" in params:
        if params["normalize"] == 1:
            norm_ = Normalizer()
            train_X = norm_.fit_transform(train_X)
            test_X = norm_.transform(test_X)
        del params["normalize"]
    if "scale" in params:
        if params["scale"] == 1:
            scale_ = StandardScaler()
            train_X = scale_.fit_transform(train_X)
            test_X = scale_.transform(test_X)
        del params["scale"]
    return train_X, test_X


def evaluate_model(test_y, yhat_classes):

    scores = {
        "MCC": matthews_corrcoef(test_y, yhat_classes),
        "F1 score": f1_score(test_y, yhat_classes, average="micro"),
        "Balanced Accuracy": balanced_accuracy_score(test_y, yhat_classes),
        "Accuracy": accuracy_score(test_y, yhat_classes),
        "Precision": precision_score(test_y, yhat_classes, average="micro"),
        "Sensitivity": recall_score(test_y, yhat_classes, average="micro"),
    }

    yhat_classes_str = ",".join(str(x) for x in yhat_classes)
    test_y_str = ",".join(str(x) for x in test_y)

    scores["test_y_pred"] = yhat_classes_str
    scores["test_y_true"] = test_y_str

    data_scores = pd.DataFrame([scores])

    return data_scores, scores["F1 score"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-mn", "--model_name", default="RFC", help="model_name")
    parser.add_argument("-tr", "--train_dataset", default="ISL", help="train_dataset")
    parser.add_argument("-te", "--test_dataset", default="ISL", help="test_dataset")
    args = parser.parse_args()
    return args


def to_labels(eq_labels, pred_probs, threshold):

    pred_labels = []


    for i in range(len(eq_labels)):
        if pred_probs[i] >= threshold:
            pred_labels.append(eq_labels[i])
        else:
            if eq_labels[i] == 0:
                pred_labels.append(1)
            else:
                pred_labels.append(0)

    return pred_labels


def evaluate_EQ_predicttion(test_y, test_EQ_LABELS, test_EQ_NUM, pred):

    # Creating combined dataframe
    df_EQ_out = pd.concat(
        [pd.DataFrame(i) for i in [test_y, test_EQ_LABELS, test_EQ_NUM, pred]], axis=1
    )
    df_EQ_out.columns = ["test_y", "EQ_LABELS", "EQ_NUM", "pred"]
    # Creating boolean matrix then grouping and taking the mean
    df_EQ_out = (
        (df_EQ_out["test_y"] == df_EQ_out["pred"])
        .groupby([df_EQ_out["EQ_LABELS"], df_EQ_out["EQ_NUM"]])
        .mean()
        .rename("pred_proba")
        .reset_index()
    )
    # Convert to string
    df_out_st = df_EQ_out.astype(str).agg(",".join).to_frame().T
    df_out_st.columns = ["EQ_true", "EQ_NUM", "EQ_pred_proba"]
    df_out_st.drop(["EQ_NUM"], axis=1)

    # # calculate the g-mean for each threshold

    # gmeans = np.sqrt(tpr * (1 - fpr))
    # # locate the index of the largest g-mean
    # ix = np.argmax(gmeans)
    #
    # # # get the best threshold  Youden’s J statistic
    # # fpr, tpr, thresholds = roc_curve(df_EQ_out['EQ_LABELS'], df_EQ_out['pred_proba'])
    # J = tpr - fpr
    # ix_J = np.argmax(J)
    #
    # best_threshold = thresholds[ix]
    # best_threshold_J = thresholds[ix_J]
    #
    # mcc = matthews_corrcoef(
    #     df_EQ_out["EQ_LABELS"],
    #     to_labels(df_EQ_out["EQ_LABELS"], df_EQ_out["pred_proba"], best_threshold),
    # )
    #
    # mcc_J = matthews_corrcoef(
    #     df_EQ_out["EQ_LABELS"],
    #     to_labels(df_EQ_out["EQ_LABELS"], df_EQ_out["pred_proba"], best_threshold_J),
    # )

    # Optimal Threshold Tuning
    thresholds = np.arange(0.7, 1, 0.01)

    # evaluate each threshold
    scores_optimal = [
        matthews_corrcoef(
            df_EQ_out["EQ_LABELS"],
            to_labels(df_EQ_out["EQ_LABELS"], df_EQ_out["pred_proba"], t),
        )
        for t in thresholds
    ]

    # get best threshold
    ix = np.argmax(scores_optimal)
    best_threshold = thresholds[ix]

    # Find prediction to the dataframe applying threshold
    yhat_classes = to_labels(
        df_EQ_out["EQ_LABELS"], df_EQ_out["pred_proba"], best_threshold
    )

    yhat_classes_str = ",".join(str(x) for x in yhat_classes)
    df_out_st["EQ_pred_classes"] = yhat_classes_str

    # calculate metrics
    scores = {
        "EQ_MCC": matthews_corrcoef(df_EQ_out["EQ_LABELS"], yhat_classes),
        "EQ_F1": f1_score(df_EQ_out["EQ_LABELS"], yhat_classes),
        "EQ_Accuracy": accuracy_score(df_EQ_out["EQ_LABELS"], yhat_classes),
        "EQ_Precision": precision_score(df_EQ_out["EQ_LABELS"], yhat_classes),
        "EQ_Sensitivity": recall_score(df_EQ_out["EQ_LABELS"], yhat_classes),
    }

    # calculate pr auc and roc auc
    precision_, recall_, _ = precision_recall_curve(
        df_EQ_out["EQ_LABELS"], df_EQ_out["pred_proba"]
    )
    auprc = auc(recall_, precision_)
    rocauc = roc_auc_score(df_EQ_out["EQ_LABELS"], df_EQ_out["pred_proba"])
    scores["EQ_PR_AUC"] = auprc
    scores["EQ_ROC_AUC"] = rocauc

    EQ_scores = pd.DataFrame([scores])
    EQ_result = pd.concat([EQ_scores, df_out_st], axis=1)

    return EQ_result, scores["EQ_MCC"]


def files_to_save(model_name_, train_dataset_, test_dataset_):

    project_root = get_project_root()

    log_file = project_root / (
        "output/"
        + model_name_
        + "_"
        + train_dataset_
        + "_"
        + test_dataset_
        + "_logfiles.log"
    )
    out_file_csv = project_root / (
        "output/"
        + model_name_
        + "_"
        + train_dataset_
        + "_"
        + test_dataset_
        + "_results.csv"
    )
    metrics_csv = project_root / (
        "output/"
        + model_name_
        + "_"
        + train_dataset_
        + "_"
        + test_dataset_
        + "_metrics.csv"
    )
    out_file_pkl = project_root / (
        "output/"
        + model_name_
        + "_"
        + train_dataset_
        + "_"
        + test_dataset_
        + "_test_pred_results.pkl"
    )
    model_save_file = project_root / (
        "output/"
        + model_name_
        + "_"
        + train_dataset_
        + "_"
        + test_dataset_
        + "_model.h5"
    )

    return log_file, out_file_csv, metrics_csv, out_file_pkl, model_save_file


def get_mail_info():

    global now_time, msg_from, passwd, msg_to, s

    now_time = datetime.datetime.strftime(
        datetime.datetime.today(), "%Y-%m-%d %H:%M:%S:%f"
    )  # 当前时间
    msg_from = "3193612728@qq.com"  # 发件人邮箱
    # passwd = "Qwert12345"  # 发件人邮箱密码
    passwd = "jlvyfubgpulldece"
    msg_to = "3193612728@qq.com"  # 收件人邮箱
    s = smtplib.SMTP_SSL("smtp.qq.com", 465)  # 发件箱邮件服务器及端口号

    return now_time, msg_from, passwd, msg_to, s


def send_err_email(name, ex, ex_detail):
    """
    :param name:程序名
    :param ex: 异常名
    :param ex_detail: 异常详情
    :return:
    """

    now_time, msg_from, passwd, msg_to, s = get_mail_info()

    subject = "【程序异常提醒】{name}-{date}".format(name=name, date=now_time)  # 标题
    content = """<div class="emailcontent" style="width:100%;max-width:720px;text-align:left;margin:0 auto;padding-top:80px;padding-bottom:20px">
        <div class="emailtitle">
            <h1 style="color:#fff;background:#51a0e3;line-height:70px;font-size:24px;font-weight:400;padding-left:40px;margin:0">程序运行异常通知</h1>
            <div class="emailtext" style="background:#fff;padding:20px 32px 20px">
                <p style="color:#6e6e6e;font-size:13px;line-height:24px">程序：<span style="color:red;">【{name}】</span>运行过程中出现异常错误，下面是具体的异常信息，请及时核查处理！</p>
                <table cellpadding="0" cellspacing="0" border="0" style="width:100%;border-top:1px solid #eee;border-left:1px solid #eee;color:#6e6e6e;font-size:16px;font-weight:normal">
                    <thead>
                        <tr>
                            <th colspan="2" style="padding:10px 0;border-right:1px solid #eee;border-bottom:1px solid #eee;text-align:center;background:#f8f8f8">爬虫异常详细信息</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td style="padding:10px 0;border-right:1px solid #eee;border-bottom:1px solid #eee;text-align:center;width:100px">异常简述</td>
                            <td style="padding:10px 20px 10px 30px;border-right:1px solid #eee;border-bottom:1px solid #eee;line-height:30px">{ex}</td>
                        </tr>
                        <tr>
                            <td style="padding:10px 0;border-right:1px solid #eee;border-bottom:1px solid #eee;text-align:center">异常详情</td>
                            <td style="padding:10px 20px 10px 30px;border-right:1px solid #eee;border-bottom:1px solid #eee;line-height:30px">{ex_detail}</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
    </div>
        """.format(
        ex=ex, ex_detail=ex_detail, name=name
    )  # 正文
    msg = MIMEText(content, _subtype="html", _charset="utf-8")
    msg["Subject"] = subject
    msg["From"] = "程序小助手<3193612728@qq.com>"
    msg["To"] = msg_to

    try:
        s.login(msg_from, passwd)
        s.sendmail(msg_from, msg_to, msg.as_string())
        print("程序异常发送成功", end="\n")
    except smtplib.SMTPException as e:
        print("程序异常发送失败", end="\n")
    finally:
        s.quit()


def send_finish_email(name):
    """
    :param name:程序名
    :return:
    """
    now_time, msg_from, passwd, msg_to, s = get_mail_info()

    subject = "【程序运行结束】{name}-{date}".format(name=name, date=now_time)  # 标题
    content = "【程序运行结束】{name}-{date}".format(name=name, date=now_time)  # 正文
    msg = MIMEText(content, _subtype="html", _charset="utf-8")
    msg["Subject"] = subject
    msg["From"] = "程序小助手<3193612728@qq.com>"
    msg["To"] = msg_to

    try:
        s.login(msg_from, passwd)
        s.sendmail(msg_from, msg_to, msg.as_string())
        print("程序运行结束发送成功", end="\n")
    except smtplib.SMTPException as e:
        print("程序运行结束发送失败", end="\n")
    finally:
        s.quit()
