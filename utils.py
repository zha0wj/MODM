import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import pickle
import datetime
from main_model import indicator, Euclidean_distance
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
import sklearn.neighbors._base
import sys
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
# from missingpy import MissForest
from sklearn.impute import KNNImputer
import shap
import matplotlib.pyplot as plt
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
filename = "./record_df" + "/result_val" + current_time + ".txt"
filename1 = "./record_df" + "/result_ext" + current_time + ".txt"
filename2 = "./record_df" + "/fill_metric" + current_time + ".txt"

alpha = torch.tensor((0.2, 0.8))
# floss = FocalLoss(class_num=2, alpha=alpha)
loss_forward = []


def train(
    model,
    model1,
    config,
    train_loader,
    valid_loader=None,
    valid_epoch_interval=5,
    foldername="",
):
    optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)  # weight_decay的作用是L2正则化，和Adam并无直接关系。
    optimizer1 = Adam(model1.parameters(), lr=1e-4, weight_decay=1e-6)

    if foldername != "":
        output_path = foldername + "/model.pth"

    p1 = int(0.4 * config["epochs"])
    p2 = int(0.8 * config["epochs"])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1, p2], gamma=0.9
    )
    lr_scheduler1 = torch.optim.lr_scheduler.MultiStepLR(
        optimizer1, milestones=[p1, p2], gamma=0.1
    )

    gain_parameters = {'batch_size': 256,
                'hint_rate': 0.95,
                'alpha': 100,
                'iterations': 10000}
    best_valid_loss = 1e10

    flag = 3 # default 0
    max_auc = 0.
    for epoch_no in range(config["epochs"]):
        avg_loss = 0.
        fl_lose = 0.
        mse_loss = 0.
        model.train()
        model1.train()

        predict_all = []
        predict_proba = []
        label_all = []

        initial_all = []
        point_all = []
        produce_all = []
        train_csdi = []
        test_csdi = []

        original_all = []         
        score_all = []         
        status_all = []
        test_point = []
        label_all = []
        
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):

                optimizer.zero_grad()
                optimizer1.zero_grad()
                loss0, generated_data, gt_point, status, original_data, imputed_point = model(train_batch)
                loss1 = (((((generated_data - original_data) * gt_point) ** 2) * 2).sum() / gt_point.sum()).\
                    requires_grad_(True)
                imputed_data = (generated_data * (1 - imputed_point) + original_data * imputed_point).view(-1, 49)

                data_im = (generated_data * (1 - imputed_point) + original_data * imputed_point).view(-1, 49)
                train_csdi.append(data_im)
                original_all.append(original_data)
                score_all.append(imputed_point)
                status_all.append(status)


                # MODM
                imputed_data[:, 41] = Euclidean_distance(imputed_data[:, 41:43], 2)     # 计算one-hot与标签的欧氏距离，并转换为label
                imputed_data[:, 42] = Euclidean_distance(imputed_data[:, 43:], 6)       # 计算one-hot与标签的欧氏距离，并转换为label
                imputed_data = imputed_data[:, :43]

                x_num = imputed_data[:, :39].to("cuda")
                x_cat = imputed_data[:, 39:].type(torch.LongTensor).to("cuda")
                imputed_data = torch.cat([x_num, x_cat], dim=1)

                output = model1(imputed_data)

                # tabnet
                # output, _ = model1(torch.squeeze(imputed_data, 1))

                # GRU
                # output = model1(torch.squeeze(imputed_data, 1))

                # TabTransformer
                # x_categ = imputed_data[:, 41:].type(torch.LongTensor).to("cuda")
                # x_cont = imputed_data[:, :41].to("cuda")
                # output = model1(torch.squeeze(x_categ, 1), torch.squeeze(x_cont, 1))

                # loss1 = ssloss(output, status)
                loss2 = torch.nn.CrossEntropyLoss()(output, status)
                status_oh = torch.zeros(output.size()).to("cuda")
                status_oh.scatter_(1, status.view((status.shape[0], 1, * (status.shape)[1:])), 1)
                loss3 = nn.MSELoss()(output, status_oh)

                t_mse = 0.8 * loss1 + 0.2 * loss3
                # loss2 = floss(output, status)
                # loss = loss0 + loss1 + loss2


                # loss = loss0 + loss1 + loss2 + loss3
                # loss = loss0


                loss = loss2.log() + (t_mse.log() + loss0.log()) / 2
                # loss = loss2.log() + (1/2 * (t_mse*t_mse + loss0*loss0)).log() / 2
                # loss = 100 * loss0 + loss1

                loss.backward()

                avg_loss += loss0.item()
                mse_loss += t_mse.item()
                fl_lose += loss2.item()

                optimizer1.step()
                optimizer.step()
                it.set_postfix(
                    ordered_dict={
                        "avg_df_loss": avg_loss / batch_no,
                        "avg_fl_loss": fl_lose / batch_no,
                        "mse_loss": mse_loss / batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=False,
                )
            lr_scheduler1.step()
            lr_scheduler.step()
            
        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            model.eval()
            model1.eval()
            avg_df_valid = 0
            avg_res_valid = 0
            avg_mse_valid = 0

            with torch.no_grad():
                
                with tqdm(valid_loader, mininterval=5.0, maxinterval=50.0) as it:
                    for batch_no, valid_batch in enumerate(it, start=1):
                        lose0, produce_data, gtv_point, label, initial_data, filled_point = model(valid_batch, is_train=0)

                        valid_data = (produce_data * (1 - filled_point) + initial_data * filled_point).view(-1, 49)
                        lose1 = (((((produce_data - initial_data) * gtv_point) ** 2) * 2).sum() / gtv_point.sum())
                        data_vl = (produce_data * (1 - filled_point) + initial_data * filled_point).view(-1, 49)

                        test_csdi.append(data_vl)
                        initial_all.append(initial_data)
                        point_all.append(gtv_point)
                        produce_all.append(produce_data)
                        test_point.append(filled_point)
                        label_all.append(label)

                        # MODM
                        valid_data[:, 41] = Euclidean_distance(valid_data[:, 41:43], 2)     # 计算one-hot与标签的欧氏距离，并转换为label
                        valid_data[:, 42] = Euclidean_distance(valid_data[:, 43:], 6)       # 计算one-hot与标签的欧氏距离，并转换为label
                        valid_data = valid_data[:, :43]

                        x_num_v = valid_data[:, :39].to("cuda")
                        x_cat_v = valid_data[:, 39:].type(torch.LongTensor).to("cuda")
                        valid_data = torch.cat([x_num_v, x_cat_v], dim=1)
                        predict = model1(valid_data)

                        # tabnet
                        # predict, _ = model1(torch.squeeze(valid_data, 1))

                        # GRU
                        # predict = model1(torch.squeeze(valid_data, 1))

                        # TabTransformer
                        # x_categv = valid_data[:, 41:].type(torch.LongTensor).to("cuda")
                        # x_contv = valid_data[:, :41].to("cuda")
                        # predict = model1(torch.squeeze(x_categv, 1), torch.squeeze(x_contv, 1))

                        lose2 = nn.CrossEntropyLoss()(predict, label)
                        # lose = lose0 + lose1
                        avg_df_valid += lose0.item()
                        avg_res_valid += lose2.item()
                        avg_mse_valid += lose1.item()
                        avg_loss_valid = avg_df_valid + avg_res_valid + avg_mse_valid

                        predict1 = np.argmax(predict.cpu().detach(), axis=1)

                        predict_all.append(predict)
                        predict_proba.append(predict1)

                        it.set_postfix(
                            ordered_dict={
                                "valid_df_loss": avg_df_valid / batch_no,
                                "valid_res_loss": avg_res_valid / batch_no,
                                "mse_loss": avg_mse_valid / batch_no,
                                "epoch": epoch_no,
                            },
                            refresh=False,
                        )

                initial_all = torch.squeeze(torch.cat(tuple(initial_all), 0), -1)
                point_all = torch.squeeze(torch.cat(tuple(point_all), 0), -1)
                produce_all = torch.squeeze(torch.cat(tuple(produce_all), 0), -1)
                test_point = torch.squeeze(torch.cat(tuple(test_point), 0), -1)
                label_all_np = torch.cat(tuple(label_all), 0).cpu().detach()

                rmse_csdi = torch.sqrt((((produce_all - initial_all) * point_all) ** 2).sum()) / point_all.sum()
                mae_csdi = ((torch.abs(produce_all - initial_all) * point_all).sum()) / point_all.sum()
                print("\ncsdi:", rmse_csdi.item(), mae_csdi.item())
            predict_all = torch.cat(tuple(predict_all), 0)
            label_all = torch.cat(tuple(label_all), 0)
            predict_proba = torch.cat(tuple(predict_proba), 0)

            fpr1,tpr1,_ = roc_curve(label_all.cpu().detach(),predict_all.cpu().detach()[:,1])
            rocc = np.concatenate((fpr1,tpr1),axis=0)
            # pd.DataFrame(rocc).to_csv("roc_res_test.csv")
            print("roc filled!")

            cm = confusion_matrix(label_all.cpu().detach(), predict_proba.cpu().detach())

            print("\n Valid confusion matrix:\n", cm)

            TN = cm[0][0]
            FP = cm[0][1]
            FN = cm[1][0]
            TP = cm[1][1]
            acc, sens, spec = indicator(TN, FP, FN, TP)
            print("\n accuracy:", acc, "\n sensitivity:", sens, "\n specificity", spec)

            auc = roc_auc_score(label_all.cpu().detach(), predict_all.cpu().detach()[:, 1])
            print("\n auc:", auc)

            with open(
                    filename, "a"
            ) as f:
                f.writelines(
                    ["\n\nepoch:", str(epoch_no), "\tdf_loss:",str(avg_df_valid / batch_no), "\tmse_loss:",str(avg_res_valid / batch_no), "\tce_loss:",str(avg_mse_valid / batch_no),
                     "\n ", "confusion matrix:", str(cm.reshape(1, -1).tolist()), " ",
                     "\n ", "AUC: ", str(auc), " ",
                     "\n ", "acc: ", str(acc), " ",
                     "\n ", "sens: ", str(sens), " ",
                     "\n ", "spec: ", str(spec), "\n\n"])

            if best_valid_loss > avg_loss_valid:
                best_valid_loss = avg_loss_valid
                print(
                    "\n best loss is updated to ",
                    avg_loss_valid / batch_no,
                    "at",
                    epoch_no,
                )
                loss_forward.append(best_valid_loss)

            train_csdi = torch.cat(tuple(train_csdi), 0).cpu().numpy()
            test_csdi = torch.cat(tuple(test_csdi), 0).cpu().numpy()
            if epoch_no == (config["epochs"] - 1) :
                
                # pd.DataFrame(train_csdi).to_csv("train_csdi.csv")
                # pd.DataFrame(test_csdi).to_csv("test_csdi.csv")
                print("over csdi fill!")

            if auc > max_auc:
                max_auc = auc
                # pd.DataFrame(train_csdi).to_csv("train_MODM.csv")
                # pd.DataFrame(test_csdi).to_csv("test_MODM.csv")
                print("over max_csdi fill!")
                print("max auc:",auc," at", epoch_no)

            

    if foldername != "":
        torch.save(model.state_dict(), output_path)


def quantile_loss(target, forecast, q: float, eval_points) -> float:
    return 2 * torch.sum(
        torch.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q))
    )


def calc_denominator(target, eval_points):
    return torch.sum(torch.abs(target * eval_points))


def calc_quantile_CRPS(target, forecast, eval_points, mean_scaler, scaler):
    target = target * scaler + mean_scaler
    forecast = forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = []
        for j in range(len(forecast)):
            q_pred.append(torch.quantile(forecast[j : j + 1], quantiles[i], dim=1))
        q_pred = torch.cat(q_pred, 0)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)


def normalization(tmp_values, tmp_masks):
    """Normalize data in [0, 1] range.

    Args:
      - data: original data

    Returns:
      - norm_data: normalized data
      - norm_parameters: min_val, max_val for each feature for renormalization
    """

    # Parameters
    _, dim = tmp_values.shape
    # norm_data = data.copy()

    mean = np.zeros(dim)
    std = np.zeros(dim)

    # For each dimension
    for k in range(dim):
        c_data = tmp_values[:, k][tmp_masks[:, k] == 1]
        mean[k] = c_data.mean()
        std[k] = c_data.std()
    norm_data = (
            (tmp_values - mean) / std * tmp_masks
    )

    # Return norm_parameters for renormalization
    norm_parameters = {'mean': mean,
                       'std': std}

    return norm_data, norm_parameters


def renormalization(norm_data, norm_parameters):
    """Renormalize data from [0, 1] range to the original range.

    Args:
      - norm_data: normalized data
      - norm_parameters: min_val, max_val for each feature for renormalization

    Returns:
      - renorm_data: renormalized original data
    """

    mean = norm_parameters['mean']
    std = norm_parameters['std']

    _, dim = norm_data.shape
    renorm_data = norm_data.copy()

    for i in range(dim):
        renorm_data[:, i] = renorm_data[:, i] * std[i] + mean[i]

    return renorm_data


def rounding(imputed_data, data_x):
    """Round imputed data for categorical variables.

    Args:
      - imputed_data: imputed data
      - data_x: original data with missing values

    Returns:
      - rounded_data: rounded imputed data
    """

    _, dim = data_x.shape
    rounded_data = imputed_data.copy()

    for i in range(dim):
        temp = data_x[~np.isnan(data_x[:, i]), i]
        # Only for the categorical variable
        if len(np.unique(temp)) < 20:
            rounded_data[:, i] = np.round(rounded_data[:, i])

    return rounded_data


def test(
    model,
    model1,
    config,
    ext_loader,
    foldername="",
):


    avg_loss_valid = 0
    cm = np.zeros((2, 2))

    predict_all = []
    predict_proba = []
    label_all = []
    data_all=[]

    data_e=[]

    with torch.no_grad():
        model.eval()
        model1.eval()
        with tqdm(ext_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, valid_batch in enumerate(it, start=1):
                lose0, produce_data, gte_point, label, initial_data, filled_point = model(valid_batch, is_train=0)
                lose1 = (((((produce_data - initial_data) * gte_point) ** 2) * 2).sum() / gte_point.sum())

                ext_data = (produce_data * (1 - filled_point) + initial_data).view(-1, 49)
                ext_data1 = (produce_data * (1 - filled_point) + initial_data).view(-1, 49)
                data_all.append(ext_data1)

                # MODM
                ext_data[:, 41] = Euclidean_distance(ext_data[:, 41:43], 2)     # 计算one-hot与标签的欧氏距离，并转换为label
                ext_data[:, 42] = Euclidean_distance(ext_data[:, 43:], 6)       # 计算one-hot与标签的欧氏距离，并转换为label
                ext_data = ext_data[:, :43]

                x_num = ext_data[:, :39].to("cuda")
                x_cat = ext_data[:, 39:].type(torch.LongTensor).to("cuda")
                ext_data = torch.cat([x_num, x_cat], dim=1)
                predict = model1(ext_data)

                # tabnet
                # predict, _ = model1(torch.squeeze(ext_data, 1))

                # GRU
                # predict = model1(torch.squeeze(ext_data, 1))

                # TabTransformer
                # x_categ = ext_data[:, 41:].type(torch.LongTensor).to("cuda")
                # x_cont = ext_data[:, :41].to("cuda")
                # predict = model1(torch.squeeze(x_categ, 1), torch.squeeze(x_cont, 1))

                lose2 = nn.CrossEntropyLoss()(predict, label)
                lose = lose0 + lose1 + lose2
                avg_loss_valid += lose.item()

                predict1 = np.argmax(predict.cpu().detach(), axis=1)

                predict_all.append(predict)
                predict_proba.append(predict1)
                label_all.append(label)

        data_all = torch.cat(tuple(data_all), 0).type(torch.float16)
        predict_all = torch.cat(tuple(predict_all), 0)
        label_all = torch.cat(tuple(label_all), 0)
        predict_proba = torch.cat(tuple(predict_proba), 0)

        fpr2,tpr2,_ = roc_curve(label_all.cpu().detach(),predict_all.cpu().detach()[:,1])
        rocc1 = np.concatenate((fpr2,tpr2),axis=0)
        # pd.DataFrame(rocc1).to_csv("roc_res_ext.csv")
        print("roc filled!")

        cm = confusion_matrix(label_all.cpu().detach(), predict_proba.cpu().detach())
        print("\n Ext confusion matrix:\n", cm)

        TN = cm[0][0]
        FP = cm[0][1]
        FN = cm[1][0]
        TP = cm[1][1]
        acc, sens, spec = indicator(TN, FP, FN, TP)
        print("\n accuracy:", acc, "\n sensitivity:", sens, "\n specificity", spec)

        auc = roc_auc_score(label_all.cpu().detach(), predict_all.cpu().detach()[:, 1])
        print("\n auc:", auc)

        print("df_loss:", lose0.item(), "mse_loss:", lose1.item(), "ce_loss:", lose2.item())
        with open(
                    filename1, "a"
            ) as f:
                f.writelines(
                    ["\n\next_data:", "\tdf_loss:",str(lose0), "\tmse_loss:",str(lose1), "\tce_loss:",str(lose2),
                     "\n ", "confusion matrix:", str(cm.reshape(1, -1).tolist()), " ",
                     "\n ", "AUC: ", str(auc), " ",
                     "\n ", "acc: ", str(acc), " ",
                     "\n ", "sens: ", str(sens), " ",
                     "\n ", "spec: ", str(spec), "\n\n"])
                
        ext_csdi = data_all.cpu().numpy()
        ext_label = label_all.cpu().numpy()
        pd.DataFrame(ext_csdi).to_csv("ext_MODM.csv")
        pd.DataFrame(ext_label).to_csv("ext_MODM_label.csv")
        print("ext_csdi fill over!")
                

    
    # shap.initjs()
    # background = data_all[np.random.choice(data_all.shape[0],1000, replace=False)]
    # explainer = shap.DeepExplainer(model1, data_all)
    # shap_values = explainer.shap_values(data_all)  # 传入特征矩阵X，计算SHAP值
    # # # 可视化第一个样本预测的解释
    # shap.summary_plot(shap_values[0],data_all.cpu().numpy(), feature_names)
    # plt.savefig('shap_summary_plot.jpg')
    # plt.close()
    #
    # # shap.force_plot(explainer.expected_value[0], shap_values[0][0,:], background[0].cpu().numpy(),feature_names, matplotlib=True)
    # shap.force_plot(explainer.expected_value[0], shap_values[0][0,:], feature_names, matplotlib=True)
    # plt.savefig('shap_force_plot.png')
    # plt.close()


def evaluate(model, test_loader, nsample=100, scaler=1, mean_scaler=0, foldername=""):

    with torch.no_grad():
        model.eval()
        mse_total = 0
        mae_total = 0
        evalpoints_total = 0

        all_target = []
        all_observed_point = []
        all_observed_time = []
        all_evalpoint = []
        all_generated_samples = []
        with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, test_batch in enumerate(it, start=1):
                output = model.evaluate(test_batch, nsample)

                samples, c_target, eval_points, observed_points = output

                # (pd.DataFrame(samples.cpu().numpy())).to_csv("./save/" + current_time + "/sample.csv", index=False)
                samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
                c_target = c_target.permute(0, 2, 1)  # (B,L,K)
                eval_points = eval_points.permute(0, 2, 1)
                observed_points = observed_points.permute(0, 2, 1)

                samples_median = samples.median(dim=1)
                all_target.append(c_target)
                all_evalpoint.append(eval_points)
                all_observed_point.append(observed_points)
                # all_observed_time.append(observed_time)
                all_generated_samples.append(samples)

                mse_current = (
                    ((samples_median.values - c_target) * eval_points) ** 2
                ) * (scaler ** 2)
                mae_current = (
                    torch.abs((samples_median.values - c_target) * eval_points) 
                ) * scaler

                mse_total += mse_current.sum().item()
                mae_total += mae_current.sum().item()
                evalpoints_total += eval_points.sum().item()

                it.set_postfix(
                    ordered_dict={
                        "rmse_total": np.sqrt(mse_total / evalpoints_total),
                        "mae_total": mae_total / evalpoints_total,
                        "batch_no": batch_no,
                    },
                    refresh=True,
                )

            with open(
                    "./" + foldername + "/generated_outputs_nsample" + str(nsample) + ".pk", "wb"
            ) as f:
                all_target = torch.cat(all_target, dim=0)
                all_evalpoint = torch.cat(all_evalpoint, dim=0)
                all_observed_point = torch.cat(all_observed_point, dim=0)
                # all_observed_time = torch.cat(all_observed_time, dim=0)
                all_generated_samples = torch.cat(all_generated_samples, dim=0)

                pickle.dump(
                    [
                        all_generated_samples,
                        all_target,
                        all_evalpoint,
                        all_observed_point,
                        # all_observed_time,
                        scaler,
                        mean_scaler,
                    ],
                    f,
                )

            CRPS = calc_quantile_CRPS(
                all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler
            )

            with open(
                "./" + foldername + "/result_nsample" + str(nsample) + ".pk", "wb"
            ) as f:
                pickle.dump(
                    [
                        np.sqrt(mse_total / evalpoints_total),
                        mae_total / evalpoints_total,
                        CRPS,
                    ],
                    f,
                )
                print("RMSE:", np.sqrt(mse_total / evalpoints_total))
                print("MAE:", mae_total / evalpoints_total)
                print("CRPS:", CRPS)

