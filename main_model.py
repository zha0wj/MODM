import numpy as np
import torch
import torch.nn as nn
from diff_models import diff_CSDI
import math
import torch.nn.functional as F
from statsmodels.stats.proportion import proportion_confint
from sklearn.metrics import confusion_matrix, roc_curve, auc
from torch.autograd import Variable


def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class CSDI_base(nn.Module):
    def __init__(self, target_dim, config, device):
        super().__init__()
        self.device = device
        self.target_dim = target_dim

        self.emb_feature_dim = config["model"]["featureemb"]
        self.target_strategy = config["model"]["target_strategy"]

        # self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim
        self.emb_total_dim = self.emb_feature_dim
        self.emb_total_dim += 1  # for conditional mask
        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim
        )

        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim

        input_dim = 2
        self.diffmodel = diff_CSDI(config_diff, input_dim)

        # parameters for diffusion models
        self.num_steps = config_diff["num_steps"]
        if config_diff["schedule"] == "quad":
            self.beta = np.linspace(
                config_diff["beta_start"] ** 0.5, config_diff["beta_end"] ** 0.5, self.num_steps
            ) ** 2
        elif config_diff["schedule"] == "linear":
            self.beta = np.linspace(
                config_diff["beta_start"], config_diff["beta_end"], self.num_steps
            )

        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = torch.tensor(self.alpha).float().to(self.device).unsqueeze(1).unsqueeze(1)

    def get_randmask(self, observed_mask):
        rand_for_mask = torch.rand_like(observed_mask) * observed_mask
        rand_for_mask = rand_for_mask.reshape(len(rand_for_mask), -1)
        for i in range(len(observed_mask)):
            sample_ratio = np.random.rand()  # missing ratio
            num_observed = observed_mask[i].sum().item()
            num_masked = round(num_observed * sample_ratio)
            rand_for_mask[i][rand_for_mask[i].topk(num_masked).indices] = -1
        cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()
        return cond_mask

    def get_hist_mask(self, observed_mask, for_pattern_mask=None):
        if for_pattern_mask is None:
            for_pattern_mask = observed_mask
        if self.target_strategy == "mix":
            rand_mask = self.get_randmask(observed_mask)

        cond_mask = observed_mask.clone()
        for i in range(len(cond_mask)):
            mask_choice = np.random.rand()
            if self.target_strategy == "mix" and mask_choice > 0.5:
                cond_mask[i] = rand_mask[i]
            else:  # draw another sample for histmask (i-1 corresponds to another sample)
                cond_mask[i] = cond_mask[i] * for_pattern_mask[i - 1] 
        return cond_mask

    def get_side_info(self, cond_mask):
        B, K, L = cond_mask.shape

        feature_embed = self.embed_layer(
            torch.arange(self.target_dim).to(self.device)
        )  # (K,emb)
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)

        side_info = feature_embed
        side_info = side_info.permute(0, 3, 2, 1)  # (B,*,K,L)

        side_mask = cond_mask.unsqueeze(1)  # (B,1,K,L)
        side_info = torch.cat([side_info, side_mask], dim=1)

        return side_info

    def calc_loss_valid(
        self, observed_data, cond_mask, observed_mask, side_info, is_train
    ):
        loss_sum = 0
        for t in range(self.num_steps):  # calculate loss for all t
            loss = self.calc_loss(
                observed_data, cond_mask, observed_mask, side_info, is_train, set_t=t
            )
            loss_sum += loss.detach()
        return loss_sum / self.num_steps

    def calc_loss(
        self, observed_data, cond_mask, observed_mask, side_info, is_train, set_t=-1
    ):
        B, K, L = observed_data.shape
        if is_train != 1:  # for validation
            t = (torch.ones(B) * set_t).long().to(self.device)
        else:
            t = torch.randint(0, self.num_steps, [B]).to(self.device)

        current_alpha = self.alpha_torch[t]  # (B,1,1)
        noise = torch.randn_like(observed_data)
        noisy_data = (current_alpha ** 0.5) * observed_data + (1.0 - current_alpha) ** 0.5 * noise
        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask)
        predicted = self.diffmodel(total_input, side_info, t)  # (B,K,L)

        target_mask = observed_mask - cond_mask
        residual = (noise - predicted) * target_mask
        num_eval = target_mask.sum()
        loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
        return loss

    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):

        cond_obs = (cond_mask * observed_data).unsqueeze(1)
        noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)
        total_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)

        return total_input

    def impute(self, observed_data, cond_mask, side_info, n_samples):
        B, K, L = observed_data.shape

        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)

        for i in range(n_samples):

            current_sample = torch.randn_like(observed_data)

            for t in range(self.num_steps - 1, -1, -1):
                cond_obs = (cond_mask * observed_data).unsqueeze(1)
                noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
                diff_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)
                predicted = self.diffmodel(diff_input, side_info, torch.tensor([t]).to(self.device))
                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                current_sample = coeff1 * (current_sample - coeff2 * predicted)

                if t > 0:
                    noise = torch.randn_like(current_sample)
                    sigma = (
                        (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                    ) ** 0.5
                    current_sample += sigma * noise

            imputed_samples[:, i] = current_sample.detach()
        return imputed_samples

    def forward(self, batch, is_train=1):
        (
            observed_data,
            observed_mask,
            gt_mask,
            for_pattern_mask,
            _,
            status,
            # paras,
        ) = self.process_data(batch)
        # loss_z = 0.
        if is_train == 0:
            cond_mask = gt_mask
        elif self.target_strategy != "random":
            cond_mask = self.get_hist_mask(
                observed_mask, for_pattern_mask=for_pattern_mask
            )
        else:
            cond_mask = self.get_randmask(observed_mask)

        side_info = self.get_side_info(cond_mask)

        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid
        generated_data = self.impute(observed_data, cond_mask, side_info, 3).to(self.device)
        generated_data_median = generated_data.permute(0, 1, 3, 2)
        generated_data_median = torch.median(generated_data_median, dim = 1).values

        # loss_c = self.pred(observed_data, cond_mask, side_info, status) if is_train == 1 else loss_z

        # return loss_func(observed_data, cond_mask, observed_mask, side_info, is_train) + loss_c
        return loss_func(observed_data, cond_mask, observed_mask, side_info, is_train), \
               generated_data_median.permute(0, 2, 1), (observed_mask-cond_mask), status, observed_data, observed_mask

    def evaluate(self, batch, n_samples):
        (
            observed_data,
            observed_mask,
            gt_mask,
            _,
            cut_length,
            _,
        ) = self.process_data(batch)

        with torch.no_grad():
            cond_mask = gt_mask
            target_mask = observed_mask - cond_mask

            side_info = self.get_side_info(cond_mask)

            samples = self.impute(observed_data, cond_mask, side_info, n_samples)

            for i in range(len(cut_length)):  # to avoid double evaluation
                target_mask[i, ..., 0 : cut_length[i].item()] = 0
        return samples, observed_data, target_mask, observed_mask


class TSB_eICU(CSDI_base):
    def __init__(self, config, device, target_dim=49): # eicu 39; physionet 110; (离散)eicu 41; physionet 114; (one-hot)eicu 47; physionet ?;
        super(TSB_eICU, self).__init__(target_dim, config, device)

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()
        status = batch["status"].to(self.device).long()
        # paras = batch["norm_parameters"].to(self.device).float()

        observed_data = observed_data.unsqueeze(-1)
        observed_mask = observed_mask.unsqueeze(-1)
        gt_mask = gt_mask.unsqueeze(-1)

        # observed_data = observed_data.permute(0, 2, 1)
        # observed_mask = observed_mask.permute(0, 2, 1)
        # gt_mask = gt_mask.permute(0, 2, 1)

        cut_length = torch.zeros(len(observed_data)).long().to(self.device)
        for_pattern_mask = observed_mask

        return (
            observed_data,
            observed_mask,
            gt_mask,
            for_pattern_mask,
            cut_length,
            status,
            # paras,
        )


class Bottlrneck(torch.nn.Module):
    def __init__(self, In_channel, Med_channel, Out_channel, downsample=False):
        super(Bottlrneck, self).__init__()
        self.stride = 1
        self.dropout = torch.nn.Dropout(0.5)
        if downsample == True:
            self.stride = 2

        self.layer = torch.nn.Sequential(
            torch.nn.Conv1d(In_channel, Med_channel, 1, self.stride),
            torch.nn.BatchNorm1d(Med_channel),
            torch.nn.ReLU(),
            torch.nn.Conv1d(Med_channel, Med_channel, 3, padding=1),
            torch.nn.BatchNorm1d(Med_channel),
            torch.nn.ReLU(),
            torch.nn.Conv1d(Med_channel, Out_channel, 1),
            torch.nn.BatchNorm1d(Out_channel),
            torch.nn.ReLU(),
        )

        if In_channel != Out_channel:
            self.res_layer = torch.nn.Conv1d(In_channel, Out_channel, 1, self.stride)
        else:
            self.res_layer = None

    def forward(self, x):
        if self.res_layer is not None:
            residual = self.res_layer(x)
        else:
            residual = x
        return self.dropout(self.layer(x)) + residual


class ResNet(nn.Module):
    def __init__(self, in_channels=1, classes=2):
        super(ResNet, self).__init__()
        self.features = torch.nn.Sequential(
            # torch.nn.Conv1d(in_channels, 32, kernel_size=7, stride=1, padding=3),
            Conv1d_with_init(in_channels, 32, kernel_size=7),
            torch.nn.MaxPool1d(3, 2, 1),

            Bottlrneck(32, 64, 64, False),
            Bottlrneck(64, 64, 256, False),
            Bottlrneck(256, 64, 256, False),
            #
            Bottlrneck(256, 128, 512, True),
            Bottlrneck(512, 128, 512, False),
            Bottlrneck(512, 128, 512, False),
            Bottlrneck(512, 128, 512, False),
            # torch.nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=3),
            Conv1d_with_init(512, 512, kernel_size=3),

            Bottlrneck(512, 256, 1024, True),
            Bottlrneck(1024, 256, 1024, False),
            Bottlrneck(1024, 512, 1024, False),

            torch.nn.AdaptiveAvgPool1d(1)
        )
        self.classifer = torch.nn.Sequential(
            torch.nn.Linear(1024, classes)
        )
        self.restrict = torch.nn.Sequential(
            torch.nn.Softmax()
        )

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = self.features(x)
        x = x.view(-1, 1024)
        x = self.classifer(x)
        # x = self.restrict(x)
        return x


class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.
            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])
        The losses are averaged across observations for each minibatch.
        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """

    def __init__(self, class_num, alpha, gamma=3, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)
        # P = inputs

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        # print(class_mask)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P * class_mask).sum(1).view(-1, 1)

        log_p = probs.log()
        # print('probs size= {}'.format(probs.size()))
        # print(probs)

        batch_loss = (-alpha * (torch.pow((1 - probs), self.gamma)) * log_p).double()
        # print('-----bacth_loss------')
        # print(batch_loss)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


def ACC(TN, FP, FN, TP):
    Acc = (TP + TN) / (TP + FN + FP + TN)
    C1acc, C2acc = proportion_confint(TP + TN, TP + FN + FP + TN, 0.05)
    return Acc, C1acc, C2acc


def SENS(TN, FP, FN, TP):
    Sens = TP / (TP + FN)
    Q1sens = 2 * TP + 1.96 * 1.96
    Q2sens = 1.96 * math.sqrt(1.96 * 1.96 + 4 * TP * FN / (TP + FN))
    Q3sens = 2 * (TP + FN + 1.96 * 1.96)
    C1sens = (Q1sens - Q2sens) / Q3sens
    C2sens = (Q1sens + Q2sens) / Q3sens
    return Sens, C1sens, C2sens


def SPEC(TN, FP, FN, TP):
    Spec = TN / (TN + FP)
    Q1spec = 2 * TN + 1.96 * 1.96
    Q2spec = 1.96 * math.sqrt(1.96 * 1.96 + 4 * FP * TN / (FP + TN))
    Q3spec = 2 * (FP + TN + 1.96 * 1.96)
    C1spec = (Q1spec - Q2spec) / Q3spec
    C2spec = (Q1spec + Q2spec) / Q3spec
    return Spec, C1spec, C2spec


def indicator(TN, FP, FN, TP):
    accuracy, c1accuracy, c2accuracy = ACC(TN, FP, FN, TP)
    sensitivity, c1sensitivity, c2sensitivity = SENS(TN, FP, FN, TP)
    specificity, c1specificity, c2specificity = SPEC(TN, FP, FN, TP)
    return accuracy, sensitivity, specificity


def Euclidean_distance(c_data, o):
    n, m = c_data.shape
    # x_min = torch.zeros(n, m)
    if o == 2:
        x_11 = torch.tensor((1, 1)).to("cuda")
        x = ((c_data - x_11) ** 2).to("cuda")
        d_data = torch.argmin(x, dim=1)
    if o == 6:
        x_11 = torch.tensor((1, 1, 1, 1, 1, 1)).to("cuda")
        x = ((c_data - x_11) ** 2).to("cuda")
        d_data = torch.argmin(x, dim=1)
    return d_data



