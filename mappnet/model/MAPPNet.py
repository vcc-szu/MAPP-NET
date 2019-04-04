from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import torch
import torch.nn as nn
import etw_pytorch_utils as pt_utils
from collections import namedtuple

import numpy as np

from pointnet2.utils.pointnet2_modules import PointnetFPModule, PointnetSAModuleMSG, PointnetSAModule

from mappnet.config import config as global_config

ModelReturn = namedtuple("ModelReturn", ["preds", "loss", "acc"])
def reconstruction_loss(pred_pc, target, seg):
    """ pred_pc: BxFxNx3,
        target: BxFxNx3, 
        seg: BxN """
    #single_target = target[:,0,:,:]
    l2 = torch.norm(target - pred_pc, dim=-1)
    l2_avg = torch.mean(l2)
    return l2_avg


def mappnet_fn(model, data, epoch=0,eval=False):
    with torch.set_grad_enabled(not eval):
        pc_in = data['pc_in']
        pc_target = data['pc_target']
        seg = data['seg']

        pc_in = pc_in.to("cuda", non_blocking=True)
        pc_target = pc_target.to("cuda", non_blocking=True)
        seg = seg.to("cuda", non_blocking=True)

        preds = model(pc_in)
        loss = mappnet_model.reconstruction_loss(preds, pc_target[:,0,:,:]-pc_in, seg)
        #_, classes = torch.max(preds, -1)
        acc = 0#(classes == labels).float().sum() / labels.numel()
        #if eval==True:
    return ModelReturn(preds, loss, {"loss": loss.item()})
def model_fn_decorator(train_set, test_set, config):
    ModelReturn = namedtuple("ModelReturn", ["preds", "loss", "acc"])
    def mappnet_fn(model, data, epoch=0,eval=False):
        with torch.set_grad_enabled(not eval):
            pc_in = data['pc_in']
            pc_target = data['pc_target']
            seg = data['seg']

            pc_in = pc_in.to("cuda", non_blocking=True)
            pc_target = pc_target.to("cuda", non_blocking=True)
            seg = seg.to("cuda", non_blocking=True)

            batch_size, num_frame, num_point, pc_channel = pc_target.size()

            pc_in_tiled =   pc_in.view(batch_size,1,num_point,pc_channel)  \
                                 .repeat(1,num_frame,1,1)
            disp_target = pc_target - pc_in_tiled
            
            preds = model(pc_in)
            loss = reconstruction_loss(preds, disp_target, seg)
            #_, classes = torch.max(preds, -1)
            acc = 0#(classes == labels).float().sum() / labels.numel()
            if eval==True:
                indices = data['index']
                for i in range(batch_size):
                    savename = test_set.get_name(indices[i])
                    for j in range(num_frame):
                        path = '{}/{}_{}.pts'.format(config.predict_save_path, savename, j+1)
                        out_pc = pc_in_tiled[i,j,:,:]+preds[i,j,:,:]
                        np.savetxt(path, out_pc.detach().cpu().numpy())
        return ModelReturn(preds, loss, {"loss": loss.item()})
    return mappnet_fn

'''
class MAPPNet(nn.Module):
    def __init__(self, num_classes, input_channels=6, use_xyz=True):
        super(MAPPNet, self).__init__()

        self.pt2SA = PointnetSAModuleMSG(num_classes, input_channels=0, use_xyz=True)
        self.lstm = nn.LSTM()
'''
class MAPPNet(nn.Module):
    r"""
        PointNet2 with multi-scale grouping
        Semantic segmentation network that uses feature propogation layers

        Parameters
        ----------
        num_classes: int
            Number of semantics classes to predict over -- size of softmax classifier that run for each point
        input_channels: int = 6
            Number of input channels in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        use_xyz: bool = True
            Whether or not to use the xyz position of a point as a feature
    """

    def __init__(self, num_classes, input_channels=6, use_xyz=True, config=global_config):
        super(MAPPNet, self).__init__()
        self.config = config

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
            .conv1d(num_classes, activation=None)
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
                (B, N, 3 + input_channels) tensor
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
        #print(pc_frames.size())
        return pc_frames

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
