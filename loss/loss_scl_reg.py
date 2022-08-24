import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(
        self,
        temperature=0.07,
        contrast_mode="one",
        base_temperature=0.07,
        gamma1=2,
        gamma2=2,
        threshold=0.5,
    ):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.threshold = threshold

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = torch.device("cuda") if features.is_cuda else torch.device("cpu")

        contrast_count = 1
        contrast_feature_smiles = features[:, 1, :]
        contrast_feature_graph = features[:, 0, :]

        mask_init = mask
        #######################################################################
        ############################anchor graph###############################
        if self.contrast_mode == "one":
            anchor_feature = contrast_feature_graph
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError("Unknown mode: {}".format(self.contrast_mode))

        # anchor graph contrast SMILES-------------------------------------------
        batch_size = features.shape[0]
        if labels is not None and mask_init is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask_init is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature_smiles.T), self.temperature
        )
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        weight = torch.sqrt(
            (torch.pow(labels.repeat(1, batch_size) - labels.repeat(1, batch_size).T, 2))
        )
        dynamic_t = torch.quantile(weight, 0.5, dim=1)
        dynamic_t = torch.where(dynamic_t > self.threshold, self.threshold, dynamic_t.double())
        mask = torch.le(weight, dynamic_t.repeat([128, 1]).T).int()

        gamma1 = self.gamma1
        gamma2 = self.gamma2

        n_weight = -weight / dynamic_t
        n_weight = 1 + torch.exp(n_weight * gamma1)
        d_weight = (
            (weight - dynamic_t.repeat([128, 1]).T).T / (torch.max(weight, dim=1)[0] - dynamic_t)
        ).T + gamma2
        d_weight = torch.exp(d_weight)

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        ) * (1 - mask)

        mask = mask

        exp_logits = torch.exp(logits) * d_weight * logits_mask
        log_prob = torch.log(torch.exp(logits * n_weight * mask)) - torch.log(
            exp_logits.sum(1, keepdim=True)
        )

        numerator = (mask * log_prob).sum(1)
        denominator = mask.sum(1)
        numerator = numerator[denominator > 0]
        denominator = denominator[denominator > 0]
        mean_log_prob_pos = numerator / denominator

        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss_graph_smiles = loss.view(anchor_count, -1).mean()

        # -----------------------------------------------------------------------
        # anchor SMILES contrast graph------------------------------------------
        batch_size = features.shape[0]
        if labels is not None and mask_init is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask_init is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature_graph.T), self.temperature
        )
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        weight = torch.sqrt(
            (torch.pow(labels.repeat(1, batch_size) - labels.repeat(1, batch_size).T, 2))
        )
        mask = torch.le(weight, dynamic_t.repeat([128, 1]).T).int()

        n_weight = -weight / dynamic_t
        n_weight = 1 + torch.exp(n_weight * gamma1)
        d_weight = (
            (weight - dynamic_t.repeat([128, 1]).T).T / (torch.max(weight, dim=1)[0] - dynamic_t)
        ).T + gamma2
        d_weight = torch.exp(d_weight)

        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        ) * (1 - mask)

        # compute log_prob
        exp_logits = torch.exp(logits) * d_weight * logits_mask
        log_prob = torch.log(torch.exp(logits * n_weight * mask)) - torch.log(
            exp_logits.sum(1, keepdim=True)
        )

        # compute mean of log-likelihood over positive
        numerator = (mask * log_prob).sum(1)
        denominator = mask.sum(1)
        numerator = numerator[denominator > 0]
        denominator = denominator[denominator > 0]
        mean_log_prob_pos = numerator / denominator

        # loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss_smiles_graph = loss.view(anchor_count, -1).mean()

        loss = loss_smiles_graph + loss_graph_smiles
        return loss
