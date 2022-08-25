import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    # Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.

    def __init__(self, temperature=0.07, contrast_mode="one", base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model.
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = torch.device(
            "cuda") if features.is_cuda else torch.device("cpu")

        contrast_count = 1
        contrast_feature_smiles = features[:, 1, :]
        contrast_feature_graph = features[:, 0, :]

        mask_init = mask

        ############################anchor graph###############################

        anchor_feature = contrast_feature_graph
        anchor_count = 1

        # anchor graph contrast SMILES
        batch_size = features.shape[0]

        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError("Num of labels does not match num of features")
        mask = torch.eq(labels, labels.T).float().to(device)
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature,
                         contrast_feature_smiles.T), self.temperature
        )
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        numerator = (mask * log_prob).sum(1)
        denominator = mask.sum(1)
        numerator = numerator[denominator > 0]
        denominator = denominator[denominator > 0]
        mean_log_prob_pos = numerator / denominator

        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss_graph_smiles = loss.view(anchor_count, -1).mean()

        ############################anchor SMILES###############################

        anchor_feature = contrast_feature_smiles
        anchor_count = 1

        # anchor SMILES contrast graph
        batch_size = features.shape[0]

        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError("Num of labels does not match num of features")
        mask = torch.eq(labels, labels.T).float().to(device)

        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature,
                         contrast_feature_graph.T), self.temperature
        )
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

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
