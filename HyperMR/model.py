import utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import scipy.io as io

from modules.transformer import TransformerEncoder
import manifolds
from manifolds.base import Manifold
from manifolds.hyperboloid import Hyperboloid
from manifolds.poincare import PoincareBall
from manifolds.euclidean import Euclidean
from hyperbolic_utils.HypLinear import HypLinear
from hypatt_layer import HyperbolicAttentionNetwork
from utils import (
    load_files,
    save_pickle,
    fix_seed,
    print_model,
    CosineAnnealingWarmUpRestarts,
)
from dataloader import idx_vocab

class ClassEmbedding(nn.Module):
    def __init__(self, cfg, trainable=True):
        super(ClassEmbedding, self).__init__()
        idx2vocab = utils.load_files(cfg["DATASET"]["IDX2VOCAB"])
        self.n_token = len(idx2vocab)
        self.word_emb_size = cfg["MODEL"]["NUM_WORD_EMB"]

        self.emb = nn.Embedding(self.n_token, self.word_emb_size)
        weight_init = utils.load_files(cfg["DATASET"]["GLOVE"]).astype(np.float32)
        weight_mat = torch.from_numpy(weight_init)
        self.emb.load_state_dict({"weight": weight_mat})

        if not trainable:
            self.emb.weight.requires_grad = False

    def forward(self, x):
        emb = self.emb(x)
        return emb


class AnswerSelector(nn.Module):
    def __init__(self, cfg):
        super(AnswerSelector, self).__init__()
        self.av2i = utils.load_files(cfg["DATASET"]["AVOCAB2IDX"])
        self.len_avocab = len(self.av2i)

        self.glove_cands = utils.load_files(cfg["DATASET"]["GLOVE_ANS_CAND"]).astype(
            np.float32
        )
        self.glove_cands = torch.from_numpy(self.glove_cands).cuda()

    def forward(self, inputs):
        similarity = torch.matmul(inputs, self.glove_cands.transpose(0, 1))
        pred = F.log_softmax(similarity, dim=1)
        return pred


class HypergraphTransformer(nn.Module):
    def __init__(self, cfg, args):
        super(HypergraphTransformer, self).__init__()

        self.cfg = cfg
        self.args = args
        self.n_hop = args.n_hop
        print("初始化模型...")
        self.word_emb_size = cfg["MODEL"]["NUM_WORD_EMB"]
        self.max_num_hqnode = cfg["MODEL"]["NUM_MAX_QNODE"]
        self.n_hidden = cfg["MODEL"]["NUM_HIDDEN"]
        self.max_num_hknode = cfg["MODEL"]["NUM_MAX_KNODE_{}H".format(self.n_hop)]
        self.n_out = cfg["MODEL"]["NUM_OUT"]
        self.n_ans = cfg["MODEL"]["NUM_ANS"]
        self.abl_only_ga = args.abl_only_ga
        self.abl_only_sa = args.abl_only_sa


        self.manifold = getattr(manifolds, Hyperboloid)()
        curvatures = nn.Parameter(torch.Tensor([1., 1., 1., 1.]).cuda())
        self.n_layers = args.num_layers
        self.curvatures = curvatures
        self.emb_curvatures = self.curvatures[0]
        self.hyp_dropout = cfg["MODEL"]["HYP_DROP"]
        self.use_bias = cfg["MODEL"]["USE_BIAS"]
        self.q_hyplinear = HypLinear(self.manifold,
                                     in_feat_dim=self.word_emb_size * self.max_num_hqnode, # 300*3
                                     out_feat_dim=self.n_hidden,    # 256
                                     c=self.emb_curvatures,
                                     # c=self.curvatures[0],
                                     dropout=self.hyp_dropout,
                                     use_bias=self.use_bias)
        self.k_hyplinear = HypLinear(self.manifold,
                                     in_feat_dim=self.word_emb_size * self.max_num_hknode, # 300*12
                                     out_feat_dim=self.n_hidden,    # 256
                                     c=self.emb_curvatures,
                                     # c=self.curvatures[0],
                                     dropout=self.hyp_dropout,
                                     use_bias=self.use_bias)

        att_layers = []
        for i in range(self.n_layers):
            # c_in, c_out = 1., 1.
            c_in, c_out = self.curvatures[i], self.curvatures[i+1]
            att_layers.append(HyperbolicAttentionNetwork(cfg, self.manifold, c_in, c_out))
        self.layers = nn.Sequential(*att_layers)

        if "pql" in args.data_name:
            self.i2e = ClassEmbedding(cfg, False)  # pql : small dataset
        else:
            self.i2e = ClassEmbedding(cfg)

        self.q2h = torch.nn.Linear(
            self.word_emb_size * self.max_num_hqnode, self.n_hidden
        )
        self.k2h = torch.nn.Linear(
            self.word_emb_size * self.max_num_hknode, self.n_hidden
        )


        self.dropout = nn.Dropout(p=self.cfg["MODEL"]["INP_DROPOUT"])
        self.out_dropout = 0.1

        if self.args.abl_ans_fc != True:
            self.proj1 = nn.Linear(2 * self.n_hidden, self.n_hidden)
            self.proj2 = nn.Linear(self.n_hidden, self.n_out)
            self.ans_selector = AnswerSelector(cfg)
        else:
            self.proj1 = nn.Linear(2 * self.n_hidden, self.n_hidden)
            self.proj2 = nn.Linear(self.n_hidden, self.n_ans)


    def hyperbolic_emb(self, Euclidean_feat):
        # x_tan = self.manifold.proj_tan0(Euclidean_feat, self.curvatures[0])
        # x_hyp = self.manifold.expmap0(x_tan, c=self.curvatures[0])
        # x_hyp = self.manifold.proj(x_hyp, c=self.curvatures[0])
        x_tan = self.manifold.proj_tan0(Euclidean_feat, self.emb_curvatures)
        x_hyp = self.manifold.expmap0(x_tan, c=self.emb_curvatures)
        x_hyp = self.manifold.proj(x_hyp, c=self.emb_curvatures)
        return x_hyp

    def forward(self, batch):
        # self.curvatures = curvatures
        # 输入是根据跳数和最大节点数量裁剪的相关节点子图，节点之间不存在边
        # 根据注意力机制，直接为每个节点计算权重，根据节点注意力聚类形成超边

        he_ques_id = batch[0]  # [BS, 150, 3]
        he_kg_id = batch[1]

        num_batch = he_ques_id.shape[0]
        num_he_ques = he_ques_id.shape[1]
        num_he_kg = he_kg_id.shape[1]

        he_ques_emb = self.i2e(he_ques_id)
        he_kg_emb = self.i2e(he_kg_id)

        # 知识图谱可视化
        # idx_vocab(self.args, self.cfg, num_batch, he_kg_id, he_kg_emb)

        he_ques = torch.reshape(he_ques_emb, (num_batch, num_he_ques, -1))    # [batch, 15, 900]
        he_kg = torch.reshape(he_kg_emb, (num_batch, num_he_kg, -1))          # [batch, 150, 3600]
        # 过滤nan值，用0代替
        he_ques = torch.where(torch.isnan(he_ques), torch.full_like(he_ques, 0), he_ques)
        he_kg = torch.where(torch.isnan(he_kg), torch.full_like(he_kg, 0), he_kg)

        he_ques = self.q2h(he_ques)  # [batch, 15, 900] → [batch, 15, 256]
        he_kg = self.k2h(he_kg)  # [batch, 150, 3600] → [batch, 150, 256]

        ques_hypblc = self.hyperbolic_emb(he_ques.view(-1, self.n_hidden))
        kg_hypblc = self.hyperbolic_emb(he_kg.view(-1, self.n_hidden))

        kg_euc = self.manifold.proj_tan0(self.manifold.logmap0(kg_hypblc, c=self.curvatures[self.n_layers]),
                                c=self.curvatures[self.n_layers])

        # self.visualization(kg_hypblc.reshape(num_batch, num_he_kg, -1)[0])

        ques_hyplinear = self.dropout(ques_hypblc)
        kg_hyplinear = self.dropout(kg_hypblc)

        # dim transform
        ques_hyplinear = ques_hyplinear.view(num_batch, num_he_ques, -1).permute(1, 0, 2)   # [3840, 256] to [150, 256, 256]
        kg_hyplinear = kg_hyplinear.view(num_batch, num_he_kg, -1).permute(1, 0, 2)         # [38400, 256] to [15, 256, 256]

        if self.args.abl_only_ga == True:# 删
            print("abl_only_ga == True")
        else:  # 不做消融
            input = (kg_hyplinear, ques_hyplinear)
            hyp_att_k, hyp_att_q = self.layers.forward(input)

            hypblc_kg_sum = torch.mean(hyp_att_k, axis=0)
            hypblc_ques_sum = torch.mean(hyp_att_q, axis=0)


            hypblc_last_kq = torch.cat([hypblc_kg_sum, hypblc_ques_sum], dim=1)

        if self.args.abl_ans_fc == False:
            euclidean_last_kq = self.manifold.proj_tan0(self.manifold.logmap0(hypblc_last_kq, c=self.curvatures[self.n_layers]), c=self.curvatures[self.n_layers])


            output = self.proj2(
                F.dropout(
                    F.relu(self.proj1(euclidean_last_kq)),
                    p=self.out_dropout,
                    training=self.training,
                )
            )
            pred = self.ans_selector(output)
        else:
            euclidean_last_kq = self.manifold.proj_tan0(self.manifold.logmap0(hypblc_last_kq, c=self.curvatures[self.n_layers]), c=self.curvatures[self.n_layers])
            output = self.proj2(
                F.dropout(
                    F.relu(self.proj1(euclidean_last_kq)),
                    p=self.out_dropout,
                    training=self.training,
                )
            )
            pred = F.log_softmax(output, dim=1)

        loss = F.nll_loss(pred, batch[-1])

        return pred, self.curvatures

    def visualization(self, fearuers):
        tsne = TSNE(n_components=3, random_state=0)
        lastkq_3D = tsne.fit_transform(fearuers.detach().cpu().numpy())

        io.savemat('point_3D.mat', {'x': lastkq_3D[:, 0], 'y': lastkq_3D[:, 1], 'z': lastkq_3D[:, 2]})
        # io.loadmat('point_3D.mat')['point_3D']
        ax = plt.subplot(projection='3d')
        ax.scatter(lastkq_3D[:, 0], lastkq_3D[:, 1], lastkq_3D[:, 2], color='red')
        ax.set_title('Hyperbolic')
        plt.show()

