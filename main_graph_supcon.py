import os
import time
import math
import argparse
from copy import deepcopy

import numpy as np
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import tensorboard_logger as tb_logger
from torch_geometric.data import DataLoader

from models.deepgcn import SupConDeeperGCN
from models.SMILESBert import SMILESBert
from utils.evaluate import Evaluator
from utils.load_our_dataset import PygOurDataset
from utils.util import (
    AverageMeter,
    adjust_learning_rate,
    set_optimizer,
    save_model,
)

try:
    import apex
except ImportError:
    pass

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def parse_option():
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
    parser.add_argument("--data_folder", type=str, default=None, help="path to custom dataset")
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

    opt.model_name = "SupCon_{}_{}_lr1_{}_decay_{}_bsz_{}_temp_{}_trial_{}_gamma1_{}_gamma2_{}_mlp_{}_decay_{}_rate_{}".format(
        opt.dataset,
        opt.model,
        opt.learning_rate,
        opt.weight_decay,
        opt.batch_size,
        opt.temp,
        opt.trial,
        opt.gamma1,
        opt.gamma2,
        opt.mlp_layers,
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


def set_loader(opt, dn):

    rr = opt.data_folder
    train_dataset = PygOurDataset(root=rr, phase="train", dataname=dn)
    test_dataset = PygOurDataset(root=rr, phase="test", dataname=dn)
    val_dataset = PygOurDataset(root=rr, phase="valid", dataname=dn)

    return train_dataset, test_dataset, val_dataset


class Classifier(torch.nn.Sequential):
    def __init__(self, model_1, model_2, opt):
        super(Classifier, self).__init__()
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

    def forward(self, batch1, opt, phase="train"):
        if opt.classification and opt.global_feature:
            global_feature = torch.cat(
                (batch1.mgf.view(batch1.y.shape[0], -1), batch1.maccs.view(batch1.y.shape[0], -1)),
                dim=1,
            ).float()
        elif not opt.classification and opt.global_feature:
            global_feature = F.normalize(
                torch.cat(
                    (
                        batch1.mgf.view(batch1.y.shape[0], -1),
                        batch1.maccs.view(batch1.y.shape[0], -1),
                    ),
                    dim=1,
                ).float(),
                dim=1,
            )

        f1_raw = self.model_1(batch1)
        f2_raw = self.model_2(
            batch1.input_ids.view(batch1.y.shape[0], -1).int(),
            batch1.attention_mask.view(batch1.y.shape[0], -1).int(),
        )
        f1_sp = self.enc1_1(f1_raw)
        f2_sp = self.enc2_1(f2_raw)

        f1_co = self.enc1_2(f1_raw)
        f2_co = self.enc1_2(f2_raw)

        f1_cross = F.normalize(self.head_1(f1_co), dim=1)
        f2_cross = F.normalize(self.head_2(f2_co), dim=1)

        f1_recon = self.recon_1(f1_sp + f1_co)
        f2_recon = self.recon_2(f2_sp + f2_co)

        h = torch.stack((f1_sp, f2_sp, f1_co, f2_co), dim=0)
        h = self.transformer_encoder(h)

        if opt.global_feature:
            h = torch.cat((h[0], h[1], h[2], h[3], global_feature), dim=1)
            o = self.fusion_global(h)
        else:
            h = torch.cat((h[0], h[1], h[2], h[3]), dim=1)
            o = self.fusion(h)

        if phase == "train":
            return (
                f1_cross,
                f2_cross,
                f1_recon,
                f2_recon,
                o,
                f1_sp,
                f2_sp,
                f1_co,
                f2_co,
                f1_raw,
                f2_raw,
            )
        else:
            return o, f1_sp, f2_sp, f1_cross, f2_cross, h


def set_model(opt):
    model_1 = SupConDeeperGCN(
        num_tasks=opt.num_tasks, mlp_layers=opt.mlp_layers, num_gc_layers=opt.num_gc_layers
    )
    model_2 = SMILESBert(num_tasks=opt.num_tasks)
    model = Classifier(model_1, model_2, opt)

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

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        model = model.cuda()
        criterion_scl = criterion_scl.cuda()
        cudnn.benchmark = False
    return model, criterion_scl, criterion_mse, criterion_task


def train(
    train_dataset,
    model,
    criterion_scl,
    criterion_mse,
    criterion_task,
    optimizer,
    epoch,
    opt,
    mu=0,
    std=0,
):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses_task = AverageMeter()
    losses_recon = AverageMeter()
    losses_scl = AverageMeter()
    losses = AverageMeter()
    train_dataset = train_dataset.shuffle()
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, drop_last=True)
    end = time.time()

    logit = []
    for step, batch in enumerate(tqdm(train_loader, desc="Iteration")):
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
            o,
            f1_sp,
            f2_sp,
            f1_co,
            f2_co,
            f1_raw,
            f2_raw,
        ) = model(batch, opt)
        features_cross = torch.cat([f1_cross.unsqueeze(1), f2_cross.unsqueeze(1)], dim=1)

        loss_task_tmp = 0
        loss_scl_tmp = 0
        loss_recon_tmp = 0
        total_num = 0

        loss_recon = (criterion_mse(f1_recon, f1_raw) + criterion_mse(f2_recon, f2_raw)) / 2.0
        for i in range(labels.shape[1]):
            is_labeled = batch.y[:, i] == batch.y[:, i]
            loss_task = criterion_task(o[is_labeled, i].squeeze(), labels[is_labeled, i].squeeze())
            loss_scl = criterion_scl(features_cross[is_labeled], labels[is_labeled, i])

            loss_task_tmp = loss_task_tmp + loss_task

            if opt.classification:
                if torch.sum(labels[is_labeled, i], dim=0) < labels.shape[0]:
                    wk = torch.sum(labels[is_labeled, i], dim=0) / labels.shape[0]
                    wk = (1 - wk) * (1 - wk)
                    loss_scl_tmp = loss_scl_tmp + wk * loss_scl
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


def validation(val_dataset, model, optimizer, epoch, opt, mu=0, std=0, save_feature=0):
    model.eval()

    if opt.classification:
        evaluator = Evaluator(name=opt.dataset, num_tasks=opt.num_tasks, eval_metric="rocauc")
    else:
        evaluator = Evaluator(name=opt.dataset, num_tasks=opt.num_tasks, eval_metric="rmse")
    val_loader = DataLoader(
        val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers
    )

    with torch.no_grad():
        y_true = []
        y_pred = []
        y_pred_1 = []
        if save_feature:
            feature_smiles = []
            feature_graph = []
            feature_smiles_sp = []
            feature_graph_sp = []
            feature = []
        for step, batch in enumerate(tqdm(val_loader, desc="Iteration")):
            batch = batch.to("cuda")
            o, f1_sp, f2_sp, f1_co, f2_co, h = model(batch, opt, "valid")

            if not opt.classification:
                o = o * std + mu
            if save_feature:
                feature_smiles.append(f2_co.detach().cpu())
                feature_graph.append(f1_co.detach().cpu())
                feature_smiles_sp.append(f2_sp.detach().cpu())
                feature_graph_sp.append(f1_sp.detach().cpu())
                feature.append(h.detach().cpu())

            y_true.append(batch.y.detach().cpu())
            y_pred.append(o.detach().cpu())

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
            auroc = evaluator.eval(input_dict)["rocauc"]
        else:
            auroc = evaluator.eval(input_dict)["rmse"]
    if save_feature:
        feature_smiles = np.concatenate(feature_smiles)
        feature_graph = np.concatenate(feature_graph)
        feature_smiles_sp = np.concatenate(feature_smiles_sp)
        feature_graph_sp = np.concatenate(feature_graph_sp)
        feature = np.concatenate(feature)

        return (
            auroc,
            feature_smiles,
            feature_graph,
            y_true,
            y_pred,
            feature_smiles_sp,
            feature_graph_sp,
            feature,
        )
    else:
        return auroc


def calmean(dataset):
    yy = []
    for i in range(len(dataset)):
        yy.append(dataset[i].y)

    return torch.mean(torch.Tensor(yy)).to("cuda"), torch.std(torch.Tensor(yy)).to("cuda")


def main():

    for dn in [opt.dataset + "_1", opt.dataset + "_2", opt.dataset + "_3"]:

        # build data loader
        train_dataset, test_dataset, val_dataset = set_loader(opt, dn)

        if opt.classification:
            mu, std = 0, 0
        else:
            mu, std = calmean(train_dataset)

        model_name_before = opt.model_name

        # build model and criterion
        model, criterion_scl, criterion_mse, criterion_task = set_model(opt)

        # build optimizer
        optimizer = set_optimizer(opt, model)
        opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
        if not os.path.isdir(opt.tb_folder):
            os.makedirs(opt.tb_folder)

        opt.save_folder = os.path.join(opt.model_path, opt.model_name)
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
                epoch,
                opt,
                mu,
                std,
            )
            time2 = time.time()
            print("epoch {}, total time {:.2f}".format(epoch, time2 - time1))

            acc = validation(val_dataset, model, optimizer, epoch, opt, mu, std)

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
                    test_acc = validation(test_dataset, model, optimizer, epoch, opt, mu, std)
                    logger.log_value("test auroc", test_acc, epoch)
                    print("test auroc:{}".format(test_acc))
                    print("val auroc:{}".format(acc))
            else:
                if acc < best_acc:
                    best_acc = acc
                    best_model = deepcopy(model).cpu()
                    best_epoch = epoch
                    test_acc = validation(test_dataset, model, optimizer, epoch, opt, mu, std)
                    logger.log_value("test rmse", test_acc, epoch)
                    print("test rmse:{}".format(test_acc))
                    print("val rmse:{}".format(acc))

        # save the last model
        print("best epoch : {}".format(best_epoch))
        save_file = os.path.join(opt.save_folder, "last_" + str(best_epoch) + ".pth")
        save_model(best_model, optimizer, opt, opt.epochs, save_file)

        (
            test_acc,
            feature_smiles,
            feature_graph,
            y_true,
            y_pred,
            feature_smiles_sp,
            feature_graph_sp,
            feature,
        ) = validation(
            test_dataset, best_model.cuda(), optimizer, epoch, opt, mu, std, save_feature=1
        )
        (
            train_acc,
            train_smiles,
            train_graph,
            _,
            _,
            train_smiles_sp,
            train_graph_sp,
            train_feature,
        ) = validation(
            train_dataset, best_model.cuda(), optimizer, epoch, opt, mu, std, save_feature=1
        )
        (
            val_acc,
            val_smiles,
            val_graph,
            _,
            _,
            val_smiles_sp,
            val_graph_sp,
            val_feature,
        ) = validation(
            val_dataset, best_model.cuda(), optimizer, epoch, opt, mu, std, save_feature=1
        )
        save_file = os.path.join(opt.save_folder, "result.txt")
        """
        with open(opt.save_folder+'/feature.npy', 'wb') as f:                   
            np.save(f, {'smiles':feature_smiles,'graph':feature_graph,'label':y_true, 'y_pred':y_pred, 'train_smiles':train_smiles, 'train_graph':train_graph,'val_smiles':val_smiles, 'val_graph':val_graph, 'train_smiles_sp':train_smiles_sp, 'train_graph_sp':train_graph_sp, 'train_feature':train_feature})
        """
        txtFile = open(save_file, "w")
        txtFile.write("validation:" + str(val_acc) + "\n")
        txtFile.write("test:" + str(test_acc) + "\n")
        txtFile.write("best epoch:" + str(best_epoch) + "\n")
        txtFile.close()

        print("Val Result:{}".format(val_acc))
        print("Test Result:{}".format(test_acc))


if __name__ == "__main__":
    main()
