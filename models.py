import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import Encoder, LayerNorm, Decoder, VariationalDropout
import math
import numpy as np
import random
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence


class ContrastVAE(nn.Module):

    def __init__(self, args):
        super(ContrastVAE, self).__init__()

        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.item_encoder_mu = Encoder(args)
        self.item_encoder_logvar = Encoder(args)
        self.item_decoder = Decoder(args)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.args = args
        self.latent_dropout = nn.Dropout(args.reparam_dropout_rate)
        self.apply(self.init_weights)
        self.temperature = nn.Parameter(torch.zeros(1), requires_grad=True)

    def add_position_embedding(self, sequence):

        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        item_embeddings = self.item_embeddings(sequence)  # shape: b*max_Sq*d
        position_embeddings = self.position_embeddings(position_ids)
        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return sequence_emb  # shape: b*max_Sq*d

    def get_embedding(self, sequence):
        '''
        without position
        :param sequence:
        :return:
        '''
        sequence_emb = self.dropout(self.LayerNorm(self.item_embeddings(sequence)))
        return sequence_emb

    def extended_attention_mask(self, input_ids):
        attention_mask = (input_ids > 0).long()  # used for mu, var
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64 b*1*1*max_Sq
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8 for causality
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)  # 1*1*max_Sq*max_Sq
        subsequent_mask = subsequent_mask.long()

        if self.args.cuda_condition:
            subsequent_mask = subsequent_mask.cuda()

        extended_attention_mask = extended_attention_mask * subsequent_mask  # shape: b*1*max_Sq*max_Sq
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        return extended_attention_mask

    def eps_anneal_function(self, step):

        return min(1.0, (1.0 * step) / self.args.total_annealing_step)

    def reparameterization(self, mu, logvar, step):  # vanila reparam

        std = torch.exp(0.5 * logvar)
        if self.training:
            eps = torch.randn_like(std)
            res = mu + std * eps
        else:
            res = mu + std
        return res

    def reparameterization1(self, mu, logvar, step):  # reparam without noise
        std = torch.exp(0.5 * logvar)
        return mu + std

    def reparameterization2(self, mu, logvar, step):  # use dropout

        if self.training:
            std = self.latent_dropout(torch.exp(0.5 * logvar))
        else:
            std = torch.exp(0.5 * logvar)
        res = mu + std
        return res

    def reparameterization3(self, mu, logvar, step):  # apply classical dropout on whole result
        std = torch.exp(0.5 * logvar)
        res = self.latent_dropout(mu + std)
        return res

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def encode(self, sequence_emb, extended_attention_mask):  # forward

        item_encoded_mu_layers = self.item_encoder_mu(sequence_emb, extended_attention_mask,
                                                      output_all_encoded_layers=True)

        item_encoded_logvar_layers = self.item_encoder_logvar(sequence_emb, extended_attention_mask, True)

        return item_encoded_mu_layers[-1], item_encoded_logvar_layers[-1]

    def decode(self, z, extended_attention_mask):
        item_decoder_layers = self.item_decoder(z, extended_attention_mask, output_all_encoded_layers=True)
        sequence_output = item_decoder_layers[-1]
        return sequence_output

    def forward(self, input_ids, aug_input_ids, step):
        sequence_emb = self.add_position_embedding(input_ids)  # shape: b*max_Sq*d
        extended_attention_mask = self.extended_attention_mask(input_ids)

        if self.args.latent_contrastive_learning:
            mu1, log_var1 = self.encode(sequence_emb, extended_attention_mask)
            mu2, log_var2 = self.encode(sequence_emb, extended_attention_mask)
            z1 = self.reparameterization1(mu1, log_var1, step)
            z2 = self.reparameterization2(mu2, log_var2, step)
            reconstructed_seq1 = self.decode(z1, extended_attention_mask)
            reconstructed_seq2 = self.decode(z2, extended_attention_mask)
            return reconstructed_seq1, reconstructed_seq2, mu1, mu2, log_var1, log_var2, z1, z2

        elif self.args.latent_data_augmentation:
            aug_sequence_emb = self.add_position_embedding(aug_input_ids)  # shape: b*max_Sq*d
            aug_extended_attention_mask = self.extended_attention_mask(aug_input_ids)

            mu1, log_var1 = self.encode(sequence_emb, extended_attention_mask)
            mu2, log_var2 = self.encode(aug_sequence_emb, aug_extended_attention_mask)
            z1 = self.reparameterization1(mu1, log_var1, step)
            z2 = self.reparameterization2(mu2, log_var2, step)
            reconstructed_seq1 = self.decode(z1, extended_attention_mask)
            reconstructed_seq2 = self.decode(z2, extended_attention_mask)
            return reconstructed_seq1, reconstructed_seq2, mu1, mu2, log_var1, log_var2, z1, z2

        else:  # vanilla attentive VAE
            mu, log_var = self.encode(sequence_emb, extended_attention_mask)
            z = self.reparameterization(mu, log_var, step)
            reconstructed_seq1 = self.decode(z, extended_attention_mask)
            return reconstructed_seq1, mu, log_var


class ContrastVAE_MultiInterest(ContrastVAE):
    def __init__(self, args):
        super(ContrastVAE_MultiInterest, self).__init__(args)
        self.n_interest = args.n_interest
        self.interest_prototypes = nn.Embedding(self.n_interest, args.hidden_size)  # 设定为八个兴趣方向
        self.fi1 = nn.Linear(2 * args.hidden_size, 256)
        self.fi2 = nn.Linear(256, 512)
        self.fi3 = nn.Linear(512, 256)
        self.fi_logvar = nn.Linear(256, args.hidden_size)

        self.fd_i1 = nn.Linear(2 * args.hidden_size, 512)
        self.fd_i2 = torch.nn.Sequential(
            nn.Linear(512, args.hidden_size)
        )

        # GRU
        self.gru_layers = nn.GRU(
            input_size=args.hidden_size,
            hidden_size=args.hidden_size,
            num_layers=2,
            bias=False,
            batch_first=True,
        )
        self.mlp = nn.Sequential(
            nn.Linear(args.hidden_size, args.hidden_size, bias=True),
            nn.ReLU()
        )

        self.interestLayerNorm = LayerNorm(args.hidden_size, eps=1e-12)

        self.args = args
        self.apply(self.init_weights)

    def interest_logvar(self, x):
        batch_size, n_interest, embed_size = x.shape
        x = x.reshape(-1, embed_size)
        h1 = self.dropout(F.relu(self.fi1(x)))
        h2 = self.dropout(F.relu(self.fi2(h1)))
        h3 = self.dropout(F.relu(self.fi3(h2)))
        logvar = self.fi_logvar(h3) * self.args.scale_coeff
        logvar = logvar.reshape(batch_size, n_interest, -1)
        return logvar

    def interest_encoder(self, seq_emb, mask):
        with torch.no_grad():
            w = self.interest_prototypes.weight.data.clone()
            w = F.normalize(w, dim=-1, p=2)
            self.interest_prototypes.weight.copy_(w)
            # 这里不对 item embedding 进行归一化，以防影响序列推荐模型的性能

        batch_size, n_seq, embed_size = seq_emb.shape
        interest_prototypes = self.interest_prototypes.weight.unsqueeze(0).repeat(batch_size, 1,
                                                                                  1)  # bs * n_interest * embed_size
        # Note: the complete code will be released after the paper is accepted

        # 计算引入序列信息的多兴趣
        psnl_interest_logvar = self.interest_logvar(torch.cat([interest_prototypes, psnl_interest_mu], dim=-1))
        return psnl_interest_mu, psnl_interest_logvar, intensity, probs

    def interest_encoder_temporal(self, input_ids, soft_interest_mu, intensity, probs):

        batch_size, n_seq = input_ids.shape
        mask = F.gumbel_softmax(probs.transpose(1, 2).reshape(-1, n_seq), tau=1, hard=True) # gumbel softmax 采样
        lens = torch.sum(mask, dim=-1)

        # 划分序列
        gru_item_seq = input_ids.unsqueeze(1).repeat(1, self.args.n_interest, 1).reshape(-1)[mask.reshape(-1).bool()]
        gru_item_seq_segs = torch.split(gru_item_seq, tuple(lens.long().detach().cpu().numpy()), dim=0)
        padded_seqs = pad_sequence(gru_item_seq_segs, batch_first=True)

        # 输入 GRU
        # Note: the complete code will be released after the paper is accepted

        psnl_interest_mu = 0.5*(soft_interest_mu + temporal_interest)
        psnl_interest_logvar = self.interest_logvar(torch.cat((interest_prototypes, psnl_interest_mu), dim=-1))

        psnl_interest_mu = self.dropout(self.interestLayerNorm(psnl_interest_mu))
        return psnl_interest_mu, psnl_interest_logvar

    def interest_reparameterize(self, mu, logvar, coeff=1.0):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def interest_decode(self, z):
        batch_size, n_interest, embed_size = z.shape
        z = z.reshape(-1, embed_size)
        d1 = F.relu(self.fd_i1(z))
        d2 = F.leaky_relu(self.fd_i2(d1))
        d3 = F.normalize(d2, dim=1)
        out = d3.reshape(batch_size, n_interest, -1)
        return out

    def interest_forward(self, input_ids, seq_emb, mask):
        mu, logvar, intensity, probs = self.interest_encoder(seq_emb, mask)
        if self.args.interest_temporal:
            mu, logvar = self.interest_encoder_temporal(input_ids, mu, intensity, probs)
        z = self.interest_reparameterize(mu, logvar) # sample
        # z = mu
        interest_prototypes = self.interest_prototypes.weight.unsqueeze(0).repeat(z.shape[0], 1, 1)
        interest_recon = self.interest_decode(
            torch.cat((interest_prototypes, z), -1))  # batch_size * n_interest * embed_size
        return mu, logvar, intensity, mu, probs # 测试不添加 interest decoder 会怎样
        return mu, logvar, intensity, interest_recon, probs

    def forward(self, input_ids, aug_input_ids, step):
        sequence_emb = self.get_embedding(input_ids)
        interest_mu, interest_logvar, intensity, interest_recon, probs = self.interest_forward(input_ids, sequence_emb, input_ids > 0)
        positional_sequence_emb = self.add_position_embedding(input_ids)  # shape: b*max_Sq*d
        extended_attention_mask = self.extended_attention_mask(input_ids)
        mu, log_var = self.encode(positional_sequence_emb, extended_attention_mask)
        z = self.reparameterization(mu, log_var, step)
        reconstructed_seq1 = self.decode(z, extended_attention_mask)

        if isinstance(aug_input_ids, int):
            return reconstructed_seq1, mu, log_var, interest_mu, interest_logvar, intensity, interest_recon, probs

        # data augmentation
        aug_sequence_emb = self.get_embedding(aug_input_ids)
        aug_interest_mu, aug_interest_logvar, aug_intensity, aug_interest_recon, aug_probs = self.interest_forward(aug_input_ids,
            aug_sequence_emb, aug_input_ids > 0)

        aug_positional_sequence_emb = self.add_position_embedding(aug_input_ids)  # shape: b*max_Sq*d
        aug_extended_attention_mask = self.extended_attention_mask(aug_input_ids)
        aug_mu, aug_log_var = self.encode(aug_positional_sequence_emb, aug_extended_attention_mask)
        aug_z = self.reparameterization(aug_mu, aug_log_var, step)
        reconstructed_seq2 = self.decode(aug_z, aug_extended_attention_mask)

        return reconstructed_seq1, reconstructed_seq2, mu, aug_mu, log_var, aug_log_var, z, aug_z, \
               interest_mu, interest_logvar, intensity, interest_recon, aug_interest_mu, aug_interest_logvar, \
               aug_intensity, aug_interest_recon, probs, aug_probs
