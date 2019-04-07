from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import torch
import torch.nn as nn
import torch.nn.functional as torchF
import etw_pytorch_utils as pt_utils
from collections import namedtuple

import numpy as np

from pointnet2.utils.pointnet2_modules import PointnetFPModule, PointnetSAModuleMSG, PointnetSAModule

from mappnet.config import config as global_config

ModelReturn = namedtuple("ModelReturn", ["preds", "loss", "acc"])
# def get_rec_loss(pred_pc, target, seg):
#     """ pred_pc: BxFxNx3,
#         target: BxFxNx3, 
#         seg: BxN """
#     #single_target = target[:,0,:,:]
#     l2 = torch.norm(target - pred_pc, dim=-1)
#     l2_avg = torch.mean(l2)
#     return l2_avg
def pairwise_l2_norm2_batch(x, y):
    """ x,y: Fxnx3
        
        return:
        squre_dist: (F, nump_x, nump_y)
    """
    nump_x = x.size()[1]
    nump_y = y.size()[1]
    num_frame, num_point, num_channel = x.size()

    xx = x.view(num_frame, nump_x, 1, num_channel).repeat(1,1, nump_y, 1)
    yy = y.view(num_frame, nump_y, 1, num_channel).repeat(1,1, nump_x, 1)
    yy = yy.permute(0, 2, 1, 3) # yy = y'

    diff = xx - yy # x - y'
    square_diff = diff ** 2
    square_dist = torch.sum(square_diff, dim=-1)

    return square_dist
def get_chamfer_loss(pred_pc, target_pc, target_seg):
    """ pred_pc: BxFxNx3,
        target_pc: BxFxNx3, 
        target_seg: BxN """
    batch_size, num_frame, num_point, _  = pred_pc.size()
    seg_mask = (target_seg==1)

    chamferLoss = 0
    #densityLoss = 0
    for k in range(batch_size):
        pred_pc_n =  pred_pc[k,:,seg_mask[k],:]
        target_pc_n = target_pc[k,:,seg_mask[k],:]
        # calculate chamfer loss
        square_dist = pairwise_l2_norm2_batch(target_pc_n, pred_pc_n)
        dist = torch.sqrt(square_dist)
        minRow, _ = torch.min(dist, dim=-1) # min yi to x
        minCol, _ = torch.min(dist, dim=-2) # min xi to y
        chamferLoss += ( torch.sum(minRow) + torch.sum(minCol) ) / num_frame

        # calculate density loss
        # square_dist2 = pairwise_l2_norm2_batch(target_pc_n, target_pc_n)
        # dist2 = torch.sqrt(square_dist2)
        # knndis = tf.nn.top_k(tf.negative(dist), k=8)
        # knndis2 = tf.nn.top_k(tf.negative(dist2), k=8)
        # densityLoss += torch.mean(torch.abs(knndis.values - knndis2.values))
    
    chamferLoss /= batch_size
    #densityLoss /= batch_size
    #loss_chamfer = (shapeLoss + densityLoss) * num_frame
    #tf.summary.scalar('loss_chamfer', loss_chamfer)
    #tf.add_to_collection('losses', loss_chamfer)
    return chamferLoss#, shapeLoss, densityLoss

def get_rec_loss(pred_disp, target_disp, target_seg):
    """ pred_disp: BxFxNx3,
        target_disp: BxFxNx3, 
        target_seg: BxN """
    batch_size, num_frame, num_point, _ = pred_disp.size()
    
    target_seg = target_seg.type(torch.cuda.FloatTensor)
    target_seg_reverse = 1.0 - target_seg
    num_mov = torch.sum(target_seg, dim=-1)
    num_ref = torch.sum(target_seg_reverse, dim=-1)

    l2 = torch.norm(target_disp-pred_disp, dim=-1)
    perframe_mov_loss = []
    perframe_ref_loss = []
    for i in range(batch_size):
        perframe_mov_loss.append(torch.sum(l2[i]*target_seg[i], dim=-1) / num_mov[i])
        perframe_ref_loss.append(torch.sum(l2[i]*target_seg_reverse[i], dim=-1) / num_ref[i])

    perframe_mov_loss = torch.stack(perframe_mov_loss)
    perframe_ref_loss = torch.stack(perframe_ref_loss)
    mov_loss = torch.mean(perframe_mov_loss)
    ref_loss = torch.mean(perframe_ref_loss) * 10
    loss_rec = mov_loss + ref_loss

    #tf.summary.scalar('loss_rec', loss_rec)
    #tf.add_to_collection('losses', loss_rec)
    return loss_rec#, mov_loss, ref_loss, perframe_mov_loss, perframe_ref_loss

def get_seg_loss(pred_seg, target_seg):
    """ pred_seg: BxNxC,
        seg: BxN, """
    loss_seg = torchF.cross_entropy(input = pred_seg.permute(0,2,1), target = target_seg)
    #tf.summary.scalar('loss_seg', loss_seg)
    #tf.add_to_collection('losses', loss_seg)
    return loss_seg

def get_reg_loss(pred_reg, target_reg):
    """ pred_reg: Bx6,
        target_reg: Bx6, """
    loss_reg = torch.mean(torch.norm(pred_reg-target_reg, dim=-1))
    #tf.summary.scalar('loss_reg', loss_reg)
    #tf.add_to_collection('losses', loss_reg)
    return loss_reg

def get_cls_loss(pred_cls, target_cls):
    """ pred_cls: B*NUM_CLASSES,
        target_cls: B, """
    loss = torchF.cross_entropy(input = pred_cls, target = target_cls)
    loss_cls = torch.mean(loss)
    #tf.summary.scalar('loss_cls', loss_cls)
    #tf.add_to_collection('losses', loss_cls)
    return loss_cls

def get_disp_loss(input_pc, pred_pc, target_pc, target_seg, target_mo, target_cls):
    """ input_pc(tiled): BxFxNx3,
        pred_pc: BxFxNx3,
        target_pc: BxFxNx3,
        target_seg: BxN
        target_mo: Bx6,
        target_cls: B,"""
    # only keep moving part
    batch_size, num_frame, num_point, num_channel = pred_pc.size()

    # target_cls = tf.cast(target_cls, dtype=tf.float32)
    seg_mask = (target_seg==1)
    mo_pos = target_mo[:,0:3]
    mo_dir = target_mo[:,3:]
    tiled_mo_dir = mo_dir.view(batch_size, 1, -1).repeat(1, num_point, 1)
    tiled_mo_pos = mo_pos.view(batch_size, 1, -1).repeat(1, num_point, 1)

    loss_r1_sum = 0
    loss_r2_sum = 0
    loss_r3_sum = 0
    loss_t1_sum = 0
    loss_t2_sum = 0
    loss_tr_sum = 0
    for k in range(batch_size):
        loss_r1 = 0
        loss_r2 = 0
        loss_r3 = 0
        loss_t1 = 0
        loss_t2 = 0        
        for i in range(num_frame):
            """(Nmov)*3"""
            input_pc_n =  input_pc[k,i,seg_mask[k],:]
            pred_pc_n =  pred_pc[k,i,seg_mask[k],:]
            target_pc_n = target_pc[k,i,seg_mask[k],:]
            tiled_mo_dir_n =  tiled_mo_dir[k,seg_mask[k],:]
            tiled_mo_pos_n = tiled_mo_pos[k,seg_mask[k],:]
            pred_disp_n = pred_pc_n - input_pc_n
            v1 = tiled_mo_dir_n
            v2 = input_pc_n - tiled_mo_pos_n
            v3 = pred_pc_n - tiled_mo_pos_n

            ''' constrain direction perpendicular '''
            v1_norm = torchF.normalize(pred_disp_n, dim=-1)
            v2_norm = torchF.normalize(tiled_mo_dir_n, dim=-1)
            cos = torch.sum(v1_norm * v2_norm, dim=-1)
            loss_r1 += torch.mean(torch.abs(cos))
            loss_t1 += torch.mean(torch.abs(1.0 - cos))

            ''' constrain moving angle consistency '''
            lenproject1 = torch.sum(v2 * v1, dim=-1) / torch.sum(v1 * v1, dim=-1)
            lenproject2 = torch.sum(v3 * v1, dim=-1) / torch.sum(v1 * v1, dim=-1)
            lenproject1 = lenproject1.view(-1,1).repeat(1, 3)
            lenproject2 = lenproject2.view(-1,1).repeat(1, 3)
            projpoint1 = tiled_mo_pos_n + lenproject1 * v1
            projpoint2 = tiled_mo_pos_n + lenproject2 * v1
            
            cos_pred = torch.sum(torchF.normalize(input_pc_n - projpoint1, dim=-1) * torchF.normalize(pred_pc_n - projpoint2, dim=-1), dim=-1)
            mean = cos_pred.mean(dim=0)
            variance = cos_pred.std(dim=0)
            loss_r2 += variance
            
            loss_r3 += torch.mean(torch.abs(torch.norm(input_pc_n-projpoint1, dim=-1) - torch.norm(pred_pc_n-projpoint2, dim=-1)))

            movdis = pred_disp_n.norm(dim=-1)
            mean = movdis.mean(dim=0)
            var = movdis.std(dim=0)
            loss_t2 += var

        loss_r1_sum += (1.0 - (target_cls[k]==0)) * loss_r1 / num_frame
        loss_r2_sum += (1.0 - (target_cls[k]==0)) * loss_r2 / num_frame
        loss_r3_sum += (1.0 - (target_cls[k]==0)) * loss_r3 / num_frame
        loss_t1_sum += (1.0 - (target_cls[k]==1)) * loss_t1 / num_frame
        loss_t2_sum += (1.0 - (target_cls[k]==1)) * loss_t2 / num_frame
        # loss_r1_sum += target_cls[k] * loss_r1 / num_frame
        # loss_r2_sum += target_cls[k] * loss_r2 / num_frame
        # loss_r3_sum += target_cls[k] * loss_r3 / num_frame
        # loss_t1_sum += (1-target_cls[k]) * loss_t1 / num_frame
        # loss_t2_sum += (1-target_cls[k]) * loss_t2 / num_frame

        loss_tr_sum += (1.0 - (target_cls[k]==2)) * (loss_r2 + loss_r3 + loss_t2) / num_frame

    loss_r1_sum /= batch_size * 3
    loss_r2_sum /= batch_size * 3
    loss_r3_sum /= batch_size * 3
    loss_t1_sum /= batch_size * 3
    loss_t2_sum /= batch_size * 3
    loss_tr_sum /= batch_size * 3

    loss_r = loss_r1_sum + loss_r2_sum + loss_r3_sum
    loss_t = loss_t1_sum + loss_t2_sum
    loss_tr = loss_tr_sum

    loss_disp = loss_r + loss_t + loss_tr
    #tf.summary.scalar('loss_disp', loss_disp)
    #tf.add_to_collection('losses', loss_disp)
    return loss_disp#, loss_t1_sum, loss_t2_sum, loss_r1_sum, loss_r2_sum, loss_r3_sum

def get_mappnet_loss(pred_disp, input_pc, target_disp, pred_seg, target_seg, pred_cls, target_cls, pred_mo, target_mo):
    """ pred_disp BFNC, input_pc BFNC, target_disp BFNC, 
        pred_seg BNC, target_seg BN, pred_cls B, target_cls B, pred_mo B(MC), target_mo B(MC)"""
    pred_pc = pred_disp + input_pc
    target_pc=target_disp+input_pc
    loss_chamfer = get_chamfer_loss(pred_pc, target_pc, target_seg)
    loss_rec     = get_rec_loss(pred_disp, target_disp, target_seg)
    loss_disp    = get_disp_loss(input_pc, pred_pc, target_pc, target_seg, target_mo, target_cls)
    loss_cls     = get_cls_loss(pred_cls, target_cls)
    loss_seg     = get_seg_loss(pred_seg, target_seg)
    loss_reg     = get_reg_loss(pred_mo, target_mo)
    #torch.tensor(0).cuda()#
    loss = loss_chamfer + loss_rec + loss_disp + loss_cls + loss_seg + loss_reg
    return loss, loss_chamfer, loss_rec, loss_disp, loss_cls, loss_seg, loss_reg

def model_fn_decorator(train_set, test_set, config):
    ModelReturn = namedtuple("ModelReturn", ["preds", "loss", "stat"])
    generate_per_epoch = 10
    def mappnet_fn(model, data, epoch=0,eval=False):
        with torch.set_grad_enabled(not eval):
            data_term = ['input_pc', 'target_pc', 'target_seg', 'target_cls', 'target_mo']
            dataCUDA=[]
            for term in data_term:
                dataCUDA.append(data[term].to("cuda", non_blocking=True))
            input_pc, target_pc, target_seg, target_cls, target_mo = dataCUDA
            batch_size, num_frame, num_point, pc_channel = target_pc.size()
            input_pc_tiled =   input_pc.view(batch_size,1,num_point,pc_channel)  \
                                 .repeat(1,num_frame,1,1)
            target_disp = target_pc - input_pc_tiled
            
            pred_disp, pred_cls, pred_seg, pred_mo = model(input_pc)
            loss, loss_chamfer, loss_rec, loss_disp, loss_cls, loss_seg, loss_reg = \
                get_mappnet_loss(pred_disp, input_pc_tiled, target_disp, pred_seg, target_seg, pred_cls, target_cls, pred_mo, target_mo)
            #oss = get_rec_loss(pred_disp, disp_target, seg)
            #_, classes = torch.max(pred_disp, -1)
            pred_seg_acc = (pred_seg.max(dim=-1)[1] == target_seg).float().sum() / target_seg.numel()
            pred_cls_acc = (pred_cls.max(dim=-1)[1] == target_cls).float().sum() / target_cls.numel()
            
            res_dict = {"loss": loss.item(),
                        "loss_chamfer": loss_chamfer.item(),
                        "loss_rec": loss_rec.item(),
                        "loss_disp": loss_disp.item(),
                        "loss_cls": loss_cls.item(),
                        "loss_seg": loss_seg.item(),
                        "loss_mo":  loss_reg.item(),
                        "seg_acc": pred_seg_acc.item(),
                        "cls_acc": pred_cls_acc.item()}
            
            if eval==True:
                indices = data['index']
                for i in range(batch_size):
                    savename = test_set.get_name(indices[i])
                    for j in range(num_frame):
                        path = '{}/{}_{}.pts'.format(config.predict_save_path, savename, j+1)
                        out_pc = input_pc_tiled[i,j,:,:]+pred_disp[i,j,:,:]
                        np.savetxt(path, out_pc.detach().cpu().numpy())
        return ModelReturn(pred_disp, loss, res_dict)
    return mappnet_fn

class MAPPNet(nn.Module):
    r"""
        PointNet2 with multi-scale grouping
        Semantic segmentation network that uses feature propogation layers

        Parameters
        ----------
        use_xyz: bool = True
            Whether or not to use the xyz position of a point as a feature
        config:  Config=global_config
            used to configure MAPPNet
    """

    def __init__(self, use_xyz=True, config=global_config):
        super(MAPPNet, self).__init__()
        self.config = config
        input_channels = config.pc_channel
        if use_xyz==True:
            input_channels-=3

        self.SA_modules = nn.ModuleList()
        c_in = input_channels
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=1024,
                radii=[0.05, 0.1],
                nsamples=[16, 32],
                mlps=[[c_in, 16, 16, 32], [c_in, 32, 32, 64]],
                use_xyz=use_xyz,
            )
        )
        c_out_0 = 32 + 64

        c_in = c_out_0
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=256,
                radii=[0.1, 0.2],
                nsamples=[16, 32],
                mlps=[[c_in, 64, 64, 128], [c_in, 64, 96, 128]],
                use_xyz=use_xyz,
            )
        )
        c_out_1 = 128 + 128

        c_in = c_out_1
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=64,
                radii=[0.2, 0.4],
                nsamples=[16, 32],
                mlps=[[c_in, 128, 196, 256], [c_in, 128, 196, 256]],
                use_xyz=use_xyz,
            )
        )
        c_out_2 = 256 + 256

        c_in = c_out_2
        self.SA_modules.append(
            PointnetSAModule(mlp=[c_in, 256, 512, 1024], use_xyz=use_xyz)
            # PointnetSAModuleMSG(
            #    npoint=1,
            #    radii=[10],
            #    nsamples=[num_point],
            #    mlps=[[c_in, 256, 512, 1024]],
            #    use_xyz=use_xyz,
            # )
            # PointnetSAModuleMSG(
            #     npoint=16,
            #     radii=[0.4, 0.8],
            #     nsamples=[16, 32],
            #     mlps=[[c_in, 256, 256, 512], [c_in, 256, 384, 512]],
            #     use_xyz=use_xyz,
            # )
        )
        c_out_3 = 1024#512 + 512

        self.lstm = nn.LSTM(1024,1024)

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointnetFPModule(mlp=[256 + input_channels, 128, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[512 + c_out_0, 256, 256]))
        self.FP_modules.append(PointnetFPModule(mlp=[512 + c_out_1, 512, 512]))
        self.FP_modules.append(PointnetFPModule(mlp=[c_out_3 + c_out_2, 512, 512]))

        self.FC_layer = (
            pt_utils.Seq(128)
            .conv1d(128, bn=True)
            .dropout()
            .conv1d(config.pc_channel, activation=None)
        )
        self.cls_FC_layer = (
            pt_utils.Seq(1024)
            .fc(512, bn=True)
            .dropout(0.5)
            .fc(256, bn=True)
            .dropout(0.5)
            .fc(config.num_mo_types, activation=None)
        )
        # self.cls_FC_layer = (
        #     pt_utils.Seq(128)
        #     .conv1d(128, bn=True)
        #     .dropout()
        #     .conv1d(config.num_mo_types, activation=None)
        # )
        self.seg_FC_layer = (
            pt_utils.Seq(128)
            .conv1d(128, bn=True)
            .dropout()
            .conv1d(config.num_segs, activation=None)
        )
        # self.reg_FC_layer = (
        #     pt_utils.Seq(128)
        #     .conv1d(128, bn=True)
        #     .dropout()
        #     .conv1d(config.motion_param_dim, activation=None)
        # )
        self.reg_FC_layer = (
            pt_utils.Seq(1024)
            .fc(512, bn=True)
            .dropout(0.5)
            .fc(256, bn=True)
            .dropout(0.5)
            .fc(config.motion_param_dim, activation=None)
        )

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, pointcloud):
        # type: (MAPPNet, torch.cuda.FloatTensor) -> pt_utils.Seq
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + config.pc_channel) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        # xyz, features = self._break_up_pc(pointcloud)

        # l_xyz, l_features = [xyz], [features]
        # for i in range(len(self.SA_modules)):
        #     li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
        #     l_xyz.append(li_xyz)
        #     l_features.append(li_features)

        # for i in range(-1, -(len(self.FP_modules) + 1), -1):
        #     l_features[i - 1] = self.FP_modules[i](
        #         l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
        #     )

        # return self.FC_layer(l_features[0]).transpose(1, 2).contiguous()


        batch_size, num_point, channels = pointcloud.size()
        hidden_size = num_point
        xyz, features = self._break_up_pc(pointcloud)
        # Encoder
        # (B, N, 3),   (B, C, N)  # start
        # li_xyz,      li_features
        # (B, N, 3),   (B, 96, N)
        # (B, 256, 3), (B, 256, 256)
        # (B, 64, 3),  (B, 512, 64)
        # (B, 1, 3),   (B, 1024, 1)
        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            # group all layer
            if i == len(self.SA_modules)-1:
                li_xyz = torch.zeros([batch_size,1,3]).cuda()
            l_xyz.append(li_xyz)
            l_features.append(li_features)
        #lstm
        pc_frames = []
        lstm_features = [torch.zeros(l_feat.size() if l_feat is not None else 1) 
                                    for l_feat in l_features ]
        lstm_in = l_features[-1]
        lstm_hidden = ( torch.randn(1, batch_size, hidden_size).cuda(),
                        torch.randn(1, batch_size, hidden_size).cuda())
        for framei in range(self.config.num_frame):
            lstm_out, lstm_hidden = self.lstm(lstm_in.view(1,batch_size,hidden_size), lstm_hidden)
            # Decoder:
            # li_features
            # (B, 512, 64)
            # (B, 512, 256)
            # (B, 256, N)
            # (B, 128, N)
            # Conv
            # (B, N, 3)
            lstm_features[-1] = lstm_out.view(batch_size, -1, 1)
            for i in range(-1, -(len(self.FP_modules) + 1), -1):
                # if lstm_features[i] is not None:
                #     print(i, lstm_features[i].size())
                # if l_features[i-1] is not None:
                #     print(i, l_features[i-1].size())
                lstm_features[i - 1] = self.FP_modules[i](
                    l_xyz[i - 1], l_xyz[i], l_features[i - 1], lstm_features[i]
                )
            #print(0,lstm_features[0].size())
            pc_framei = self.FC_layer(lstm_features[0]).transpose(1, 2).contiguous()
            pc_frames.append(pc_framei)
            lstm_in = lstm_out
        # (B, F, N, 3)
        pc_frames = torch.stack(pc_frames, dim=1)

        pred_cls = self.cls_FC_layer(l_features[-1].squeeze(-1))
        pred_seg = self.seg_FC_layer(lstm_features[0]).transpose(1, 2).contiguous()
        pred_reg = self.reg_FC_layer(l_features[-1].squeeze(-1))
        return pc_frames, pred_cls, pred_seg, pred_reg

if __name__ == "__main__":
    from torch.autograd import Variable
    import numpy as np
    import torch.optim as optim

    B = 2
    N = 32
    inputs = torch.randn(B, N, 6).cuda()
    labels = torch.from_numpy(np.random.randint(0, 3, size=B * N)).view(B, N).cuda()
    model = MAPPNet(3, input_channels=3)
    model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    print("Testing with xyz")
    model_fn = model_fn_decorator(nn.CrossEntropyLoss())
    for _ in range(5):
        optimizer.zero_grad()
        _, loss, _ = model_fn(model, (inputs, labels))
        loss.backward()
        print(loss.data[0])
        optimizer.step()

    # with use_xyz=False
    inputs = torch.randn(B, N, 6).cuda()
    labels = torch.from_numpy(np.random.randint(0, 3, size=B * N)).view(B, N).cuda()
    model = MAPPNet(3, input_channels=3, use_xyz=False)
    model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    print("Testing without xyz")
    model_fn = model_fn_decorator(nn.CrossEntropyLoss())
    for _ in range(5):
        optimizer.zero_grad()
        _, loss, _ = model_fn(model, (inputs, labels))
        loss.backward()
        print(loss.data[0])
        optimizer.step()
