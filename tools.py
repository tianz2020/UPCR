import ipdb
import torch
from option import option as op
import functools

class Tools():

    @staticmethod
    def one_hot(tensor, n_vocab):
        shape = list(tensor.shape)  # [b,1]
        shape = shape + [n_vocab]  # [shape.size,n_vocab]    [b,1,n]
        new_tensor = tensor.new_zeros(*shape, dtype=torch.float)
        new_tensor = new_tensor.scatter(dim=-1,
                                        index=tensor.unsqueeze(-1),
                                        src=torch.ones_like(tensor, dtype=torch.float).unsqueeze(-1))
        return new_tensor

    @staticmethod
    def _single_decode(input_seq, src_hiddens, src_mask, decoder, input_mask=None,ret_last_step=True):
        """_single_decode"""
        batch_size = input_seq.size(0)
        trg_seq_mask = Tools.get_subsequent_mask(input_seq)  # 1, L', L'
        trg_seq_mask = trg_seq_mask.expand(batch_size, -1, -1)  # B, L', L'  参数-1就是不变，不是-1就是扩展到，这里是复制bs行
        if input_mask is not None:
            trg_seq_mask = input_mask & trg_seq_mask

        dec_output = decoder(input_seq, trg_seq_mask, src_hiddens, src_mask)  # B, L', H

        if ret_last_step:
            last_step_dec_output = dec_output[:, -1, :].unsqueeze(1)  # B, 1, H
            return last_step_dec_output
        else:
            return dec_output

    @staticmethod
    def get_subsequent_mask(seq):
        ''' For masking out the subsequent info. '''
        sz_b, len_s = seq.size(0), seq.size(1)
        subsequent_mask = (1 - torch.triu(
            torch.ones((1, len_s, len_s)), diagonal=1)).bool()
        # 1, L, L
        # 下三角矩阵
        # tensor([[[1., 0., 0.],
        #          [1., 1., 0.],
        #          [1., 1., 1.]]])
        subsequent_mask = subsequent_mask.cuda()
        return subsequent_mask

    @staticmethod
    def _generate_init(batch_size, n_vocab, trg_bos_idx, training=True):
        # B, 1
        ret = torch.ones(batch_size, 1, dtype=torch.long) * trg_bos_idx
        # B, 1, V
        if training :
            ret = Tools.one_hot(ret, n_vocab)
        ret = ret.cuda()
        return ret

    @staticmethod
    def get_mask_via_len(length, max_len):
        """"""
        B = length.size(0)  # batch size
        mask = torch.ones([B, max_len]).cuda()
        mask = torch.cumsum(mask, 1)  # [ [1,2,3,4,5..], [1,2,3,4,5..] .. ] [B,max_len]
        mask = mask <= length.unsqueeze(
            1)  # [ [True,True,..,Flase],[True,True,..,Flase],..  ] 第一个列表中True的个数为第一个session中的句子长度，后面填充的都是false
        mask = mask.unsqueeze(-2)  # [B,1,max_len]

        return mask

    @staticmethod
    def nested_index_select(origin_data, select_index):
        origin_data_shape = list(origin_data.shape)
        select_index_shape = list(select_index.shape)
        work_axes = len(select_index_shape) - 1
        grad_v = functools.reduce(lambda x, y: x * y, origin_data_shape[:work_axes])
        new_dim = select_index_shape[-1]
        grad = torch.arange(0, grad_v, dtype=torch.long).unsqueeze(-1)
        grad = grad.expand(-1, new_dim)
        grad = grad.reshape(-1)
        grad = grad * origin_data_shape[work_axes]
        select_index = select_index.reshape(-1) + grad
        reshaped_data = origin_data.reshape(grad_v * origin_data_shape[work_axes], -1)
        selected_data = reshaped_data.index_select(0, select_index)
        origin_data_shape[work_axes] = new_dim
        selected_data = selected_data.reshape(origin_data_shape)

        return selected_data

    @staticmethod
    def repeat_penalty(dist, pad_idx=None):
        """repeat penalty < 0

        we encourage the state and action distribution not overlapped, kl
        measures the distance between different distribution, thus return
        -kl.

        Args:
            dist:       B, L, V
            pad_idx:    integer
        """
        L, V = dist.size(1), dist.size(2)
        diag = torch.ones(L, dtype=torch.float)
        mask = torch.ones(L, L, dtype=torch.float) - torch.diag_embed(diag)
        mask = mask.unsqueeze(0).cuda()  # 1, L, L

        eps = 1e-9
        dist1 = dist.unsqueeze(2).expand(-1, -1, L, -1)
        dist2 = dist.unsqueeze(1).expand(-1, L, -1, -1)
        pad_mask = torch.ones(1, 1, 1, V, dtype=torch.float)
        if pad_idx is not None:
            pad_mask[:, :, :, pad_idx] = .0
        pad_mask = pad_mask.cuda()
        kl = (dist1 * torch.log(dist1 / (dist2 + eps) + eps) * pad_mask).sum(-1)  # B, L, L
        kl = (kl * mask).sum(-1).sum(-1) / (L * (L - 1))
        return - kl.mean()

    @staticmethod
    def entropy_restrain(dist):
        """entropy regularization > 0

        a lower entropy is encourage, thus return entropy.

        Args:
            dist:       B, V or B, L, V
        """
        eps = 1e-9
        if len(dist.shape) == 3:
            B, L, V = dist.shape
            dist = dist.reshape(-1, V)
        else:
            B = dist.size(1)
        entropy = (dist * torch.log(dist + eps)).sum() / B
        return - entropy

