import os
import time
import math
import argparse
from copy import deepcopy
from typing import Set, Callable, Any

import numpy as np
from tqdm import tqdm
import torch
from torch import Tensor
from torch.nn import Module
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch_geometric.data import DataLoader, Data
import tensorboard_logger as tb_logger

from models.deepgcn import SupConDeeperGCN
from models.smiles_bert import SMILESBert
from utils.evaluate import Evaluator
from utils.load_dataset import PygOurDataset
from utils.util import AverageMeter, adjust_learning_rate, set_optimizer, save_model, calmean

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def parse_option():
    """Parse arguments."""

    parser = argparse.ArgumentParser("argument for training")

    parser.add_argument("--classification", action="store_true", help="classification task")

    parser.add_argument("--wscl", type=float, default=1, help="weight of scl")
    parser.add_argument("--wrecon", type=float, default=1, help="weight of recon")

    parser.add_argument("--global_feature", action="store_true", help="with global feature")
    parser.add_argument("--batch_size", type=int, default=256, help="batch_size")
    parser.add_argument("--num_workers", type=int, default=16, help="num of workers to use")
    parser.add_argument("--epochs", type=int, default=1000, help="number of training epochs")

    # optimization
    parser.add_argument("--learning_rate", type=float, default=0.05, help="learning rate")
    parser.add_argument(
        "--lr_decay_epochs", type=str, default="1000", help="where to decay lr, can be a list"
    )
    parser.add_argument(
        "--lr_decay_rate", type=float, default=0.1, help="decay rate for learning rate"
    )
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum")

    parser.add_argument("--model", type=str, default="DeeperGCN")
    parser.add_argument("--dataset", type=str, default="freesolv", help="dataset")
    parser.add_argument("--data_dir", type=str, default=None, help="path to custom dataset")
    parser.add_argument("--num_tasks", type=int, default=1, help="parameter for task number")

    parser.add_argument("--temp", type=float, default=0.07, help="temperature for loss function")
    parser.add_argument("--gamma1", type=float, default=2)
    parser.add_argument("--gamma2", type=float, default=2)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--mlp_layers", type=int, default=2)
    parser.add_argument("--num_gc_layers", type=int, default=3)
    # other setting
    parser.add_argument("--cosine", action="store_true", help="using cosine annealing")
    parser.add_argument(
        "--syncBN", action="store_true", help="using synchronized batch normalization"
    )
    parser.add_argument("--warm", action="store_true", help="warm-up for large batch training")
    parser.add_argument("--trial", type=str, default="0", help="id for recording multiple runs")

    opt = parser.parse_args()

    opt.model_path = "./save/SupCon/{}_models".format(opt.dataset)
    opt.tb_path = "./save/SupCon/{}_tensorboard".format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(",")
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    if opt.classification:
        opt.model_name = (
            "SupCon_{}_lr_{}_bsz_{}_trial_{}_mlp_{}_wscl_{}_wrecon_{}_decay_{}_rate_{}".format(
                opt.model,
                opt.learning_rate,
                opt.batch_size,
                opt.trial,
                opt.mlp_layers,
                opt.wscl,
                opt.wrecon,
                opt.lr_decay_epochs,
                opt.lr_decay_rate,
            )
        )
    else:
        opt.model_name = "SupCon_{}_lr_{}_bsz_{}_trial_{}_gamma1_{}_gamma2_{}_mlp_{}_wscl_{}_wrecon_{}_decay_{}_rate_{}".format(
            opt.model,
            opt.learning_rate,
            opt.batch_size,
            opt.trial,
            opt.gamma1,
            opt.gamma2,
            opt.mlp_layers,
            opt.wscl,
            opt.wrecon,
            opt.lr_decay_epochs,
            opt.lr_decay_rate,
        )

    if opt.cosine:
        opt.model_name = "{}_cosine".format(opt.model_name)

    if opt.batch_size > 1024:
        opt.warm = True
    if opt.warm:
        opt.model_name = "{}_warm".format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate**3)
            opt.warmup_to = (
                eta_min
                + (opt.learning_rate - eta_min)
                * (1 + math.cos(math.pi * opt.warm_epochs / opt.epochs))
                / 2
            )
        else:
            opt.warmup_to = opt.learning_rate_gcn

    return opt


opt = parse_option()
if opt.classification:
    from loss.loss_scl_cls import SupConLoss
else:
    from loss.loss_scl_reg import SupConLoss


def set_loader(opt: Any, dataname: str) -> Set[Data]:
    """Load dataset from opt.datas_dir.

    Args:
        opt (Any): Parsed arguments.
        dataname (str): The folder name of the dataset.

    Returns:
        Set[Data]: train/validation/test sets.
    """

    train_dataset = PygOurDataset(root=opt.data_dir, phase="train", dataname=dataname)
    test_dataset = PygOurDataset(root=opt.data_dir, phase="test", dataname=dataname)
    val_dataset = PygOurDataset(root=opt.data_dir, phase="valid", dataname=dataname)

    return train_dataset, test_dataset, val_dataset


class BSCL(torch.nn.Sequential):
    """The BSCL network."""

    def __init__(self, model_1: Module, model_2: Module, opt: Any):
        """Initialization of the BSCL network.

        Args:
            model_1 (Module): The graph network
            model_2 (Module): The SMILES network
            opt (Any): Parsed arguments
        """
        super(BSCL, self).__init__()
        self.model_1 = model_1
        self.model_2 = model_2

        dim_feat = 128
        num_heads = 2

        self.enc1_1 = torch.nn.Sequential(
            torch.nn.Linear(dim_feat, dim_feat),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(dim_feat, dim_feat),
        )
        self.enc1_2 = torch.nn.Sequential(
            torch.nn.Linear(dim_feat, dim_feat),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(dim_feat, dim_feat),
        )

        self.enc2_1 = torch.nn.Sequential(
            torch.nn.Linear(dim_feat, dim_feat),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(dim_feat, dim_feat),
        )

        self.enc2_2 = torch.nn.Sequential(
            torch.nn.Linear(dim_feat, dim_feat),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(dim_feat, dim_feat),
        )

        self.head_1 = torch.nn.Sequential(
            torch.nn.Linear(dim_feat, dim_feat),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(dim_feat, dim_feat),
        )

        self.head_2 = torch.nn.Sequential(
            torch.nn.Linear(dim_feat, dim_feat),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(dim_feat, dim_feat),
        )
        self.recon_1 = torch.nn.Linear(dim_feat, dim_feat)
        self.recon_2 = torch.nn.Linear(dim_feat, dim_feat)

        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=dim_feat, nhead=num_heads)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.fusion = torch.nn.Sequential()
        self.fusion.add_module(
            "fusion_layer_1", torch.nn.Linear(in_features=dim_feat * 4, out_features=dim_feat * 2)
        )
        self.fusion.add_module("fusion_layer_1_dropout", torch.nn.Dropout(0.5))
        self.fusion.add_module("fusion_layer_1_activation", torch.nn.ReLU())
        self.fusion.add_module(
            "fusion_layer_3", torch.nn.Linear(in_features=dim_feat * 2, out_features=opt.num_tasks)
        )

        self.fusion_global = torch.nn.Sequential()
        self.fusion_global.add_module(
            "fusion_layer_1",
            torch.nn.Linear(in_features=dim_feat * 4 + 2048 + 167, out_features=dim_feat * 2),
        )
        self.fusion_global.add_module("fusion_layer_1_dropout", torch.nn.Dropout(0.5))
        self.fusion_global.add_module("fusion_layer_1_activation", torch.nn.ReLU())
        self.fusion_global.add_module(
            "fusion_layer_3", torch.nn.Linear(in_features=dim_feat * 2, out_features=opt.num_tasks)
        )

    def forward(self, batch: Tensor, opt: Any, phase: str = "train"):
        """The network of the BSCL.

        Args:
            batch1 (Tensor): Input batch
            opt (Any): Parsed arguments.
            phase (str, optional): Train phase or validation phase. Defaults to "train".

        Returns:
            Prediction results and representations learend by the model.
        """
        if opt.classification and opt.global_feature:
            global_feature = torch.cat(
                (batch.mgf.view(batch.y.shape[0], -1), batch.maccs.view(batch.y.shape[0], -1)),
                dim=1,
            ).float()
        elif not opt.classification and opt.global_feature:
            global_feature = F.normalize(
                torch.cat(
                    (
                        batch.mgf.view(batch.y.shape[0], -1),
                        batch.maccs.view(batch.y.shape[0], -1),
                    ),
                    dim=1,
                ).float(),
                dim=1,
            )

        f1_raw = self.model_1(batch)
        f2_raw = self.model_2(
            batch.input_ids.view(batch.y.shape[0], -1).int(),
            batch.attention_mask.view(batch.y.shape[0], -1).int(),
        )
        f1_sp = self.enc1_1(f1_raw)
        f2_sp = self.enc2_1(f2_raw)

        f1_co = self.enc1_2(f1_raw)
        f2_co = self.enc1_2(f2_raw)

        f1_cross = F.normalize(self.head_1(f1_co), dim=1)
        f2_cross = F.normalize(self.head_2(f2_co), dim=1)

        f1_recon = self.recon_1(f1_sp + f1_co)
        f2_recon = self.recon_2(f2_sp + f2_co)

        h_out = torch.stack((f1_sp, f2_sp, f1_co, f2_co), dim=0)
        h_out = self.transformer_encoder(h_out)

        if opt.global_feature:
            if opt.classification:
                h_out = torch.cat((h_out[0], h_out[1], h_out[2], h_out[3], global_feature), dim=1)
                output = self.fusion_global(h_out)
            else:
                h_out = torch.cat((h_out[0], h_out[1], h_out[2], h_out[3]), dim=1)
                h_out = (h_out - torch.mean(h_out)) / torch.std(h_out)
                output = self.fusion_global(torch.cat((h_out, global_feature), dim=1))
        else:
            h_out = torch.cat((h_out[0], h_out[1], h_out[2], h_out[3]), dim=1)
            output = self.fusion(h_out)

        if phase == "train":
            return (
                f1_cross,
                f2_cross,
                f1_recon,
                f2_recon,
                output,
                f1_sp,
                f2_sp,
                f1_co,
                f2_co,
                f1_raw,
                f2_raw,
            )
        else:
            return output, f1_sp, f2_sp, f1_cross, f2_cross, h_out


def set_model(opt: Any):
    """Initialization of the model and loss functions.

    Args:
        opt (Any): Parsed arguments.

    Returns:
        Return the model and the loss functions.
    """
    model_1 = SupConDeeperGCN(opt)
    model_2 = SMILESBert()
    model = BSCL(model_1, model_2, opt)

    for name, param in model.named_parameters():
        if "model_2.model.embeddings" in name or "model_2.model.encoder" in name:
            param.requires_grad = False
            print(name)

    if opt.classification:
        criterion_scl = SupConLoss(temperature=opt.temp, base_temperature=opt.temp)
    else:
        criterion_scl = SupConLoss(
            temperature=opt.temp,
            base_temperature=opt.temp,
            gamma1=opt.gamma1,
            gamma2=opt.gamma2,
            threshold=opt.threshold,
        )

    if opt.classification:
        criterion_task = torch.nn.BCEWithLogitsLoss()
    else:
        criterion_task = torch.nn.MSELoss()
    criterion_mse = torch.nn.MSELoss()

    if torch.cuda.is_available():
        model = model.cuda()
        criterion_scl = criterion_scl.cuda()
        cudnn.benchmark = False
    return model, criterion_scl, criterion_mse, criterion_task


def train(
    train_dataset: Set[Data],
    model: torch.nn.Sequential,
    criterion_scl: Callable,
    criterion_mse: Callable,
    criterion_task: Callable,
    optimizer: Optimizer,
    opt: Any,
    mu: int = 0,
    std: int = 0,
):
    """One epoch training.

    Args:
        train_dataset (Set[Data]): Train set.
        model (torch.nn.Sequential): Model
        criterion_scl (Callable): Supervised contrastive loss function
        criterion_mse (Callable): Reconstruction loss function
        criterion_task (Callable): Task loss function
        optimizer (Optimizer): Optimizer
        opt (Any): Parsed arguments
        mu (int, optional): Mean value of the train set for the regression task. Defaults to 0.
        std (int, optional): Standard deviation of the train set for the regression task.
            Defaults to 0.

    Returns:
        Losses.
    """
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses_task = AverageMeter()
    losses_recon = AverageMeter()
    losses_scl = AverageMeter()
    losses = AverageMeter()
    train_dataset_shuffle = train_dataset.shuffle()
    train_loader = DataLoader(train_dataset_shuffle, batch_size=opt.batch_size, drop_last=True)
    end = time.time()

    for _, batch in enumerate(tqdm(train_loader, desc="Iteration")):
        batch = batch.to("cuda")
        data_time.update(time.time() - end)

        bsz = batch.y.shape[0]

        if not opt.classification:
            labels = (batch.y - mu) / std
        else:
            labels = batch.y
        # compute loss
        (
            f1_cross,
            f2_cross,
            f1_recon,
            f2_recon,
            output,
            _,
            _,
            _,
            _,
            f1_raw,
            f2_raw,
        ) = model(batch, opt)
        features_cross = torch.cat([f1_cross.unsqueeze(1), f2_cross.unsqueeze(1)], dim=1)

        loss_task_tmp = 0
        loss_scl_tmp = 0
        total_num = 0

        loss_recon = (criterion_mse(f1_recon, f1_raw) + criterion_mse(f2_recon, f2_raw)) / 2.0
        for i in range(labels.shape[1]):
            is_labeled = batch.y[:, i] == batch.y[:, i]
            loss_task = criterion_task(
                output[is_labeled, i].squeeze(), labels[is_labeled, i].squeeze()
            )
            loss_scl = criterion_scl(features_cross[is_labeled], labels[is_labeled, i])

            loss_task_tmp = loss_task_tmp + loss_task

            if opt.classification:
                if torch.sum(labels[is_labeled, i], dim=0) > 0:
                    loss_scl_tmp = loss_scl_tmp + loss_scl
                    total_num = total_num + 1
            else:
                loss_scl_tmp = loss_scl_tmp + loss_scl
                total_num = total_num + 1

        if total_num == 0:
            continue

        loss_task = loss_task_tmp / labels.shape[1]
        loss_scl = loss_scl_tmp / total_num
        loss = opt.wscl * loss_scl + opt.wrecon * loss_recon + loss_task

        # update metric
        losses_task.update(loss_task.item(), bsz)
        losses_recon.update(loss_recon.item(), bsz)
        losses_scl.update(loss_scl.item(), bsz)
        losses.update(loss.item(), bsz)
        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return losses_task.avg, losses_recon.avg, losses_scl.avg, losses.avg


def validation(
    dataset: Set[Data],
    model: torch.nn.Sequential,
    opt: Any,
    mu: int = 0,
    std: int = 0,
    save_feature: int = 0,
):
    """Calculate performance metrics.

    Args:
        dataset (Set[Data]): A dataset.
        model (torch.nn.Sequential): Model.
        opt (Any): Parsed arguments.
        mu (int, optional): Mean value of the train set for the regression task.
            Defaults to 0.
        std (int, optional): Standard deviation of the train set for the regression task.
            Defaults to 0.
        save_feature (int, optional): Whether save the learned features or not.
            Defaults to 0.

    Returns:
        auroc or rmse value.
    """
    model.eval()

    if opt.classification:
        evaluator = Evaluator(name=opt.dataset, num_tasks=opt.num_tasks, eval_metric="rocauc")
    else:
        evaluator = Evaluator(name=opt.dataset, num_tasks=opt.num_tasks, eval_metric="rmse")
    data_loader = DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers
    )

    with torch.no_grad():
        y_true = []
        y_pred = []
        if save_feature:
            feature_smiles = []
            feature_graph = []
            feature_smiles_sp = []
            feature_graph_sp = []
            feature = []
        for _, batch in enumerate(tqdm(data_loader, desc="Iteration")):
            batch = batch.to("cuda")
            output, f1_sp, f2_sp, f1_co, f2_co, h_out = model(batch, opt, "valid")

            if not opt.classification:
                output = output * std + mu
            if save_feature:
                feature_smiles.append(f2_co.detach().cpu())
                feature_graph.append(f1_co.detach().cpu())
                feature_smiles_sp.append(f2_sp.detach().cpu())
                feature_graph_sp.append(f1_sp.detach().cpu())
                feature.append(h_out.detach().cpu())

            if opt.classification:
                sigmoid = torch.nn.Sigmoid()
                output = sigmoid(output)

            y_true.append(batch.y.detach().cpu())
            y_pred.append(output.detach().cpu())

        y_true = torch.cat(y_true, dim=0).squeeze().unsqueeze(1).numpy()
        if opt.num_tasks > 1:
            y_pred = np.concatenate(y_pred)
            input_dict = {"y_true": y_true.squeeze(), "y_pred": y_pred.squeeze()}
        else:
            y_pred = np.expand_dims(np.concatenate(y_pred), 1)
            input_dict = {
                "y_true": np.expand_dims(y_true.squeeze(), 1),
                "y_pred": np.expand_dims(y_pred.squeeze(), 1),
            }

        if opt.classification:
            eval_result = evaluator.eval(input_dict)["rocauc"]
        else:
            eval_result = evaluator.eval(input_dict)["rmse"]

    if save_feature:
        feature_smiles = np.concatenate(feature_smiles)
        feature_graph = np.concatenate(feature_graph)
        feature_smiles_sp = np.concatenate(feature_smiles_sp)
        feature_graph_sp = np.concatenate(feature_graph_sp)
        feature = np.concatenate(feature)

        return (
            eval_result,
            feature_smiles,
            feature_graph,
            y_true,
            y_pred,
            feature_smiles_sp,
            feature_graph_sp,
            feature,
        )
    else:
        return eval_result


def main():

    for dataname in [opt.dataset + "_1", opt.dataset + "_2", opt.dataset + "_3"]:

        # build data loader
        train_dataset, test_dataset, val_dataset = set_loader(opt, dataname)

        if opt.classification:
            mu, std = 0, 0
        else:
            mu, std = calmean(train_dataset)

        # build model and criterion
        model, criterion_scl, criterion_mse, criterion_task = set_model(opt)

        # build optimizer
        optimizer = set_optimizer(opt, model)

        model_name = "{}_{}".format(opt.model_name, dataname)

        # save folder
        opt.tb_folder = os.path.join(opt.tb_path, model_name)
        if not os.path.isdir(opt.tb_folder):
            os.makedirs(opt.tb_folder)

        opt.save_folder = os.path.join(opt.model_path, model_name)
        if not os.path.isdir(opt.save_folder):
            os.makedirs(opt.save_folder)
        # tensorboard
        logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

        if opt.classification:
            best_acc = 0
        else:
            best_acc = 10000000
        best_model = model
        best_epoch = 0

        # training routine
        for epoch in range(1, opt.epochs + 1):
            torch.cuda.empty_cache()
            adjust_learning_rate(opt, optimizer, epoch, opt.learning_rate)
            # train for one epoch
            time1 = time.time()
            loss_task, loss_recon, loss_scl, loss = train(
                train_dataset,
                model,
                criterion_scl,
                criterion_mse,
                criterion_task,
                optimizer,
                opt,
                mu,
                std,
            )
            time2 = time.time()
            print("epoch {}, total time {:.2f}".format(epoch, time2 - time1))

            acc = validation(val_dataset, model, opt, mu, std)

            # tensorboard logger
            logger.log_value("loss_task", loss_task, epoch)
            logger.log_value("loss_recon", loss_recon, epoch)
            logger.log_value("loss_scl", loss_scl, epoch)
            logger.log_value("loss", loss, epoch)
            logger.log_value("validation auroc/rmse", acc, epoch)

            if opt.classification:
                if acc > best_acc:
                    best_acc = acc
                    best_model = deepcopy(model).cpu()
                    best_epoch = epoch
                    test_acc = validation(test_dataset, model, opt, mu, std)
                    logger.log_value("test auroc", test_acc, epoch)
                    print("test auroc:{}".format(test_acc))
                    print("val auroc:{}".format(acc))
            else:
                if acc < best_acc:
                    best_acc = acc
                    best_model = deepcopy(model).cpu()
                    best_epoch = epoch
                    test_acc = validation(test_dataset, model, opt, mu, std)
                    logger.log_value("test rmse", test_acc, epoch)
                    print("test rmse:{}".format(test_acc))
                    print("val rmse:{}".format(acc))

        # save the last model
        print("best epoch : {}".format(best_epoch))
        save_file = os.path.join(opt.save_folder, "last_" + str(best_epoch) + ".pth")
        save_model(best_model, optimizer, opt, opt.epochs, save_file)

        test_acc = validation(test_dataset, best_model.cuda(), opt, mu, std)
        val_acc = validation(val_dataset, best_model.cuda(), opt, mu, std)
        save_file = os.path.join(opt.save_folder, "result.txt")

        txtFile = open(save_file, "w")
        txtFile.write("validation:" + str(val_acc) + "\n")
        txtFile.write("test:" + str(test_acc) + "\n")
        txtFile.write("best epoch:" + str(best_epoch) + "\n")
        txtFile.close()

        print("Val Result:{}".format(val_acc))
        print("Test Result:{}".format(test_acc))


if __name__ == "__main__":
    main()
