import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules.module import Module
from modules.transformer import TransformerEncoder
from hyperbolic_utils.HypLinear import HypLinear

class HyperbolicAttentionNetwork(nn.Module):
    def __init__(self, cfg, manifold, c_in, c_out):
        super(HyperbolicAttentionNetwork, self).__init__()
        self.cfg = cfg
        self.manifold = manifold
        self.n_hidden = cfg["MODEL"]["NUM_HIDDEN"]
        self.hyp_co_att = HyperbolicCoAttention(cfg, manifold, c_in, self.n_hidden)
        self.hyp_self_att = HyperbolicSelfAttention(cfg, manifold, c_in, self.n_hidden)
        # self.hyp_dropout = cfg["MODEL"]["HYP_DROP"]
        # self.use_bias = cfg["MODEL"]["USE_BIAS"]
        # self.hyplinear = HypLinear(self.manifold,
        #                              in_feat_dim=self.n_hidden * 2,  # 256*2
        #                              out_feat_dim=self.n_hidden,  # 256
        #                              c=c_in,
        #                              dropout=self.hyp_dropout,
        #                              use_bias=self.use_bias)
        self.hyp_activate = HyperbolicActivate(cfg, manifold, c_in, c_out)
    def forward(self, input):
        hyp_kg, hyp_ques = input
        hypblc_kq_coatt, hypblc_qk_coatt = self.hyp_co_att.forward(hyp_kg, hyp_ques)

        hypblc_k_satt, hypblc_q_satt = self.hyp_self_att.forward(hypblc_kq_coatt, hypblc_qk_coatt)


        hypblc_k_act = self.hyp_activate.forward(hypblc_k_satt)
        hypblc_q_act = self.hyp_activate.forward(hypblc_q_satt)

        return hypblc_k_act, hypblc_q_act

class HyperbolicCoAttention(nn.Module):
    def __init__(self, cfg, manifold, curvature, n_hidden):
        super(HyperbolicCoAttention, self).__init__()
        self.cfg = cfg
        self.manifold = manifold
        self.curvature = curvature
        self.n_hidden = n_hidden
        self.kq_coatt = TransformerEncoder(
            embed_dim=self.n_hidden,
            num_heads=self.cfg["MODEL"]["NUM_HEAD"],                # 4
            layers=self.cfg["MODEL"]["NUM_LAYER"],                  # 2
            # layers=max(self.cfg["MODEL"]["NUM_LAYER"], layers),   # 2
            attn_dropout=self.cfg["MODEL"]["ATTN_DROPOUT_K"],       # 0
            relu_dropout=self.cfg["MODEL"]["RELU_DROPOUT"],         # 0
            res_dropout=self.cfg["MODEL"]["RES_DROPOUT"],           # 0
            embed_dropout=self.cfg["MODEL"]["EMB_DROPOUT"],         # 0
            attn_mask=self.cfg["MODEL"]["ATTN_MASK"],
            fc_hid_coeff=self.cfg["MODEL"]["FC_HID_COEFF"],
        )
        self.qk_coatt = TransformerEncoder(
            embed_dim=self.n_hidden,
            num_heads=self.cfg["MODEL"]["NUM_HEAD"],                # 4
            layers=self.cfg["MODEL"]["NUM_LAYER"],                  # 2
            # layers=max(self.cfg["MODEL"]["NUM_LAYER"], layers),   # 2
            attn_dropout=self.cfg["MODEL"]["ATTN_DROPOUT_K"],       # 0
            relu_dropout=self.cfg["MODEL"]["RELU_DROPOUT"],         # 0
            res_dropout=self.cfg["MODEL"]["RES_DROPOUT"],           # 0
            embed_dropout=self.cfg["MODEL"]["EMB_DROPOUT"],         # 0
            attn_mask=self.cfg["MODEL"]["ATTN_MASK"],
            fc_hid_coeff=self.cfg["MODEL"]["FC_HID_COEFF"],
        )

    def forward(self, hypblc_kg, hypblc_ques):

        kg_size = hypblc_kg.size()
        ques_size = hypblc_ques.size()
        # H to E
        if len(kg_size) > 2 and len(ques_size) > 2:
            hypblc_kg = hypblc_kg.contiguous().view(-1, kg_size[-1])
            hypblc_ques = hypblc_ques.contiguous().view(-1, ques_size[-1])
            tangent_kg = self.manifold.logmap0(hypblc_kg, self.curvature)
            tangent_ques = self.manifold.logmap0(hypblc_ques, self.curvature)
            tangent_kg = tangent_kg.view(kg_size[0], kg_size[1], kg_size[2])
            tangent_ques = tangent_ques.view(ques_size[0], ques_size[1], ques_size[2])
        else:
            tangent_kg = self.manifold.logmap0(hypblc_kg, self.curvature)
            tangent_ques = self.manifold.logmap0(hypblc_ques, self.curvature)
        # co_att
        kq_coatt = self.kq_coatt(tangent_kg, tangent_ques, tangent_ques)
        qk_coatt = self.qk_coatt(tangent_ques, tangent_kg, tangent_kg)
        # E to H
        if len(kg_size) > 2 and len(ques_size) > 2:
            kq_coatt = kq_coatt.contiguous().view(-1, kg_size[-1])
            qk_coatt = qk_coatt.contiguous().view(-1, ques_size[-1])
            hypblc_kq_coatt = self.manifold.proj(self.manifold.expmap0(kq_coatt, c=self.curvature), c=self.curvature)
            hypblc_qk_coatt = self.manifold.proj(self.manifold.expmap0(qk_coatt, c=self.curvature), c=self.curvature)
            hypblc_kq_coatt = hypblc_kq_coatt.view(kg_size[0], kg_size[1], kg_size[2])
            hypblc_qk_coatt = hypblc_qk_coatt.view(ques_size[0], ques_size[1], ques_size[2])
        else:
            hypblc_kq_coatt = self.manifold.proj(self.manifold.expmap0(kq_coatt, c=self.curvature), c=self.curvature)
            hypblc_qk_coatt = self.manifold.proj(self.manifold.expmap0(qk_coatt, c=self.curvature), c=self.curvature)

        return hypblc_kq_coatt, hypblc_qk_coatt


class HyperbolicSelfAttention(nn.Module):
    def __init__(self, cfg, manifold, curvature, n_hidden):
        super(HyperbolicSelfAttention, self).__init__()
        self.cfg = cfg
        self.manifold = manifold
        self.curvature = curvature
        self.n_hidden = n_hidden
        self.k_satt = TransformerEncoder(
            embed_dim=self.n_hidden,
            num_heads=self.cfg["MODEL"]["NUM_HEAD"],                # 4
            layers=self.cfg["MODEL"]["NUM_LAYER"],                  # 2
            # layers=max(self.cfg["MODEL"]["NUM_LAYER"], layers),   # 2
            attn_dropout=self.cfg["MODEL"]["ATTN_DROPOUT_K"],       # 0
            relu_dropout=self.cfg["MODEL"]["RELU_DROPOUT"],         # 0
            res_dropout=self.cfg["MODEL"]["RES_DROPOUT"],           # 0
            embed_dropout=self.cfg["MODEL"]["EMB_DROPOUT"],         # 0
            attn_mask=self.cfg["MODEL"]["ATTN_MASK"],
            fc_hid_coeff=self.cfg["MODEL"]["FC_HID_COEFF"],
        )
        self.q_satt = TransformerEncoder(
            embed_dim=self.n_hidden,
            num_heads=self.cfg["MODEL"]["NUM_HEAD"],                # 4
            layers=self.cfg["MODEL"]["NUM_LAYER"],                  # 2
            # layers=max(self.cfg["MODEL"]["NUM_LAYER"], layers),   # 2
            attn_dropout=self.cfg["MODEL"]["ATTN_DROPOUT_K"],       # 0
            relu_dropout=self.cfg["MODEL"]["RELU_DROPOUT"],         # 0
            res_dropout=self.cfg["MODEL"]["RES_DROPOUT"],           # 0
            embed_dropout=self.cfg["MODEL"]["EMB_DROPOUT"],         # 0
            attn_mask=self.cfg["MODEL"]["ATTN_MASK"],
            fc_hid_coeff=self.cfg["MODEL"]["FC_HID_COEFF"],
        )
    def forward(self, hypblc_kg, hypblc_ques):
        kg_size = hypblc_kg.size()
        ques_size = hypblc_ques.size()
        # H to E
        if len(kg_size) > 2 and len(ques_size) > 2:
            hypblc_kg = hypblc_kg.contiguous().view(-1, kg_size[-1])
            hypblc_ques = hypblc_ques.contiguous().view(-1, ques_size[-1])
            tangent_kg = self.manifold.logmap0(hypblc_kg, self.curvature)
            tangent_ques = self.manifold.logmap0(hypblc_ques, self.curvature)
            tangent_kg = tangent_kg.view(kg_size[0], kg_size[1], kg_size[2])
            tangent_ques = tangent_ques.view(ques_size[0], ques_size[1], ques_size[2])
        else:
            tangent_kg = self.manifold.logmap0(hypblc_kg, self.curvature)
            tangent_ques = self.manifold.logmap0(hypblc_ques, self.curvature)
        # co_att
        k_satt = self.k_satt(tangent_kg)
        q_satt = self.k_satt(tangent_ques)
        # E to H
        if len(kg_size) > 2 and len(ques_size) > 2:
            k_satt = k_satt.contiguous().view(-1, kg_size[-1])
            q_satt = q_satt.contiguous().view(-1, ques_size[-1])
            hypblc_k_satt = self.manifold.proj(self.manifold.expmap0(k_satt, c=self.curvature), c=self.curvature)
            hypblc_q_satt = self.manifold.proj(self.manifold.expmap0(q_satt, c=self.curvature), c=self.curvature)
            hypblc_k_satt = hypblc_k_satt.view(kg_size[0], kg_size[1], kg_size[2])
            hypblc_q_satt = hypblc_q_satt.view(ques_size[0], ques_size[1], ques_size[2])
        else:
            hypblc_k_satt = self.manifold.proj(self.manifold.expmap0(k_satt, c=self.curvature), c=self.curvature)
            hypblc_q_satt = self.manifold.proj(self.manifold.expmap0(q_satt, c=self.curvature), c=self.curvature)

        return hypblc_k_satt, hypblc_q_satt

class HyperbolicActivate(nn.Module):
    def __init__(self, cfg, manifold, c_in, c_out):
        super(HyperbolicActivate, self).__init__()
        self.cfg = cfg
        self.manifold = manifold
        self.c_in = c_in
        self.c_out = c_out
    def forward(self, x):
        x_size = x.size()
        if len(x_size) > 2:
            x = x.contiguous().view(-1, x_size[-1])
            tangent_x = self.manifold.logmap0(x, c=self.c_in)
            tangent_x = tangent_x.view(x_size[0], x_size[1], x_size[2])
            activated_x = F.relu(tangent_x)
            activated_x = activated_x.contiguous().view(-1, x_size[-1])
            hypblc_activated_x = self.manifold.proj_tan0(activated_x, c=self.c_out)
            hypblc_activated_x = self.manifold.proj(self.manifold.expmap0(hypblc_activated_x, c=self.c_out), c=self.c_out)
            return hypblc_activated_x.view(x_size[0], x_size[1], x_size[2])
        else:
            xt = F.relu(self.manifold.logmap0(x, c=self.c_in))
            xt = self.manifold.proj_tan0(xt, c=self.c_out)
            return self.manifold.proj(self.manifold.expmap0(xt, c=self.c_out), c=self.c_out)

def hyperbolic_att(self, hypblc_kg, hypblc_ques):
    tangent_kg = self.manifold.logmap0(hypblc_kg, self.curvatures)
    tangent_ques = self.manifold.logmap0(hypblc_ques, self.curvatures)

    # kg
    kq_gatt = self.trans_k_with_q(tangent_kg, tangent_ques, tangent_ques)
    hypblc_kq_gatt = self.manifold.proj(self.manifold.expmap0(kq_gatt, c=self.curvatures), c=self.curvatures)

    tan_kq_gatt = self.manifold.logmap0(hypblc_kq_gatt, self.curvatures)
    k_satt = self.trans_k_mem(tan_kq_gatt)
    hypblc_k_satt = self.manifold.proj(self.manifold.expmap0(k_satt, c=self.curvatures), c=self.curvatures)

    # ques
    qk_gatt = self.trans_q_with_k(tangent_ques, tangent_kg, tangent_kg)
    hypblc_qk_gatt = self.manifold.proj(self.manifold.expmap0(qk_gatt, c=self.curvatures), c=self.curvatures)

    tan_qk_gatt = self.manifold.logmap0(hypblc_qk_gatt, self.curvatures)
    q_satt = self.trans_q_mem(tan_qk_gatt)
    hypblc_q_satt = self.manifold.proj(self.manifold.expmap0(q_satt, c=self.curvatures), c=self.curvatures)

    hypblc_kg_sum = torch.mean(hypblc_k_satt, axis=0)    # 备忘 * 10
    hypblc_ques_sum = torch.mean(hypblc_q_satt, axis=0)

    return hypblc_kg_sum, hypblc_ques_sum