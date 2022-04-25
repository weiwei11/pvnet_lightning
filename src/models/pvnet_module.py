import einops
import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import torch.nn.functional as F
from einops import rearrange

from lib.csrc.uncertainty_pnp import un_pnp_utils
from lib.csrc.ransac_voting.ransac_voting_gpu import ransac_voting_layer_v3, estimate_voting_distribution_with_mean
from src.utils.pvnet import pvnet_pose_utils
from src.utils.loss import ProxyVotingLoss


def decode_keypoint(output, un_pnp=False):
    vertex = output['vertex'].permute(0, 2, 3, 1)
    b, h, w, vn_2 = vertex.shape
    vertex = vertex.view(b, h, w, vn_2 // 2, 2)
    mask = torch.argmax(output['seg'], 1)

    if un_pnp:
        mean = ransac_voting_layer_v3(mask, vertex, 512, inlier_thresh=0.99)
        kpt_2d, var = estimate_voting_distribution_with_mean(mask, vertex, mean)
        output.update({'mask': mask, 'kpt_2d': kpt_2d, 'var': var})
    else:
        kpt_2d = ransac_voting_layer_v3(mask, vertex, 128, inlier_thresh=0.99, max_num=100)
        output.update({'mask': mask, 'kpt_2d': kpt_2d})
    return output


def uncertainty_pnp(kpt_3d, kpt_2d, var, K):
    cov_invs = []
    for vi in range(var.shape[0]):
        if var[vi, 0, 0] < 1e-6 or np.sum(np.isnan(var)[vi]) > 0:
            cov_invs.append(np.zeros([2, 2]).astype(np.float32))
        else:
            if np.linalg.det(scipy.linalg.sqrtm(var[vi])) == 0.0:  # can't compute inv
                cov_invs.append(np.zeros([2, 2]).astype(np.float32))
            else:
                cov_inv = np.linalg.inv(scipy.linalg.sqrtm(var[vi]))
                cov_invs.append(cov_inv)

    cov_invs = np.asarray(cov_invs)  # pn,2,2
    weights = cov_invs.reshape([-1, 4])
    weights = weights[:, (0, 1, 3)]
    pose_pred = un_pnp_utils.uncertainty_pnp(kpt_2d, weights, kpt_3d, K)

    return pose_pred


def solve_poses(output, ref_data, un_pnp=False):
    kpt_2d = output['kpt_2d'].detach().cpu().numpy()
    kpt_3d = ref_data['kpt_3d'].detach().cpu().numpy()
    K = ref_data['K'].detach().cpu().numpy()

    pred_poses = np.zeros((len(kpt_2d), 3, 4))
    for i in range(len(kpt_2d)):
        if un_pnp:
            var = output['var'][i].detach().cpu().numpy()
            pose_pred = uncertainty_pnp(kpt_3d[i], kpt_2d[i], var, K[i])
        else:
            pose_pred = pvnet_pose_utils.pnp(kpt_3d[i], kpt_2d[i], K[i])
        pred_poses[i] = pose_pred

    output.update({'pred_poses': pred_poses})
    return output


class PVNetLitModule(pl.LightningModule):

    def __init__(self, net: torch.nn.Module, evaluators, train_config, loss_config, un_pnp=False):
        # train_config = {lr, weight_decay, milestones, gamma}
        # loss_config = {dpvl_weight}
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.net = net
        self.un_pnp = un_pnp

        self.vote_crit = F.smooth_l1_loss
        self.seg_crit = nn.CrossEntropyLoss()
        if self.hparams.loss_config.dpvl_weight > 0:
            self.dpvl_crit = ProxyVotingLoss()

        self.val_evaluator = evaluators['val']
        self.test_evaluator = evaluators['test']

    def forward(self, x, ref_data):
        output = self.net(x)
        output = decode_keypoint(output, self.un_pnp)
        output = solve_poses(output, ref_data, self.un_pnp)
        return output

    def step(self, batch):
        output = self.net(batch['inp'])
        loss = 0

        weight = batch['mask'][:, None].float()
        vertex = batch['vertex']  # vertex is not norm
        vertex_unit = rearrange(F.normalize(rearrange(vertex, 'b (k d) h w -> b k d h w', d=2), dim=2), 'b k d h w -> b (k d) h w')

        sample_num = weight.sum()
        b, c, h, w = vertex.shape

        vote_loss = self.vote_crit(output['vertex'] * weight, vertex_unit * weight, reduction='sum')
        vote_loss = vote_loss / sample_num / c
        loss += vote_loss

        if self.hparams.loss_config.dpvl_weight > 0:
            output_vertex = output['vertex'].permute(0, 2, 3, 1).view(b, h, w, c // 2, 2)
            target_vertex = vertex.permute(0, 2, 3, 1).view(b, h, w, c // 2, 2)
            dpvl_loss = self.dpvl_crit(output_vertex * weight.view([b, h, w, 1, 1]),
                                       target_vertex * weight.view([b, h, w, 1, 1]), reduction='sum')
            dpvl_loss = self.hparams.loss_config.dpvl_weight * dpvl_loss / sample_num / c
            loss += dpvl_loss

        mask = batch['mask'].long()
        seg_loss = self.seg_crit(output['seg'], mask)
        loss += seg_loss

        loss_dict = {'vote_loss': loss, 'seg_loss': seg_loss, 'loss': loss}
        if self.hparams.loss_config.dpvl_weight > 0:
            loss_dict['dpvl_loss'] = dpvl_loss

        return loss_dict, output

    def training_step(self, batch, batch_idx):
        loss_dict, output = self.step(batch)

        for k, v in loss_dict.items():
            self.log(f'train/{k}', v, prog_bar=True)

        return {'loss': loss_dict['loss']}

    def validation_step(self, batch, batch_idx):
        loss_dict, output = self.step(batch)

        output = decode_keypoint(output, self.un_pnp)
        output = solve_poses(output, batch, self.un_pnp)

        # evaluate
        bn = len(output['pred_poses'])
        for i in range(bn):
            self.val_evaluator.evaluate(self.package(output, batch, i))

        for k, v in loss_dict.items():
            self.log(f'val/{k}', v, prog_bar=True)

        return {'loss': loss_dict['loss']}

    def validation_epoch_end(self, outputs) -> None:
        for k, v in self.val_evaluator.summarize().items():
            self.log(f'val/{k}', v, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss_dict, output = self.step(batch)

        output = decode_keypoint(output, self.un_pnp)
        output = solve_poses(output, batch, self.un_pnp)

        # evaluate
        bn = len(output['pred_poses'])
        for i in range(bn):
            self.test_evaluator.evaluate(self.package(output, batch, i))

        for k, v in loss_dict.items():
            self.log(f'test/{k}', v, prog_bar=True)

        return {'loss': loss_dict['loss']}

    def test_epoch_end(self, outputs):
        for k, v in self.test_evaluator.summarize().items():
            self.log(f'test/{k}', v, prog_bar=True)

    def configure_optimizers(self):
        lr = self.hparams.train_config.lr
        weight_decay = self.hparams.train_config.weight_decay

        params = []
        for key, value in self.net.named_parameters():
            if not value.requires_grad:
                continue
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
        optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.hparams.train_config.milestones,
                                                   gamma=self.hparams.train_config.gamma)
        return [optimizer], [scheduler]

    def package(self, output, batch, i):
        pose_pred = output['pred_poses'][i]
        pose_gt = einops.asnumpy(batch['pose'][i])
        K = einops.asnumpy(batch['K'][i])
        mask_pred = einops.asnumpy(output['mask'][i])
        mask_gt = einops.asnumpy(batch['mask'][i])
        return {
            'pose_pred': pose_pred, 'pose_gt': pose_gt, 'K': K,
            'mask_pred': mask_pred, 'mask_gt': mask_gt,
            }
