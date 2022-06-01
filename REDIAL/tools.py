import torch
import functools

class Tools():
    @staticmethod
    def one_hot(tensor, n_vocab):
        shape = list(tensor.shape)
        shape = shape + [n_vocab]
        new_tensor = tensor.new_zeros(*shape, dtype=torch.float)
        new_tensor = new_tensor.scatter(dim=-1,
                                        index=tensor.unsqueeze(-1),
                                        src=torch.ones_like(tensor, dtype=torch.float).unsqueeze(-1))
        return new_tensor

    @staticmethod
    def _single_decode(input_seq, src_hiddens, src_mask, decoder, input_mask=None,ret_last_step=True):
        batch_size = input_seq.size(0)
        trg_seq_mask = Tools.get_subsequent_mask(input_seq)
        trg_seq_mask = trg_seq_mask.expand(batch_size, -1, -1)
        if input_mask is not None:
            trg_seq_mask = input_mask & trg_seq_mask
        dec_output = decoder(input_seq, trg_seq_mask, src_hiddens, src_mask)
        if ret_last_step:
            last_step_dec_output = dec_output[:, -1, :].unsqueeze(1)
            return last_step_dec_output
        else:
            return dec_output

    @staticmethod
    def get_subsequent_mask(seq):
        sz_b, len_s = seq.size(0), seq.size(1)
        subsequent_mask = (1 - torch.triu(
            torch.ones((1, len_s, len_s)), diagonal=1)).bool()
        subsequent_mask = subsequent_mask.cuda()
        return subsequent_mask

    @staticmethod
    def _generate_init(batch_size, n_vocab, trg_bos_idx, training=True):
        ret = torch.ones(batch_size, 1, dtype=torch.long) * trg_bos_idx
        if training :
            ret = Tools.one_hot(ret, n_vocab)
        ret = ret.cuda()
        return ret

    @staticmethod
    def get_mask_via_len(length, max_len):
        """"""
        B = length.size(0)
        mask = torch.ones([B, max_len]).cuda()
        mask = torch.cumsum(mask, 1)
        mask = mask <= length.unsqueeze(1)
        mask = mask.unsqueeze(-2)
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
        L, V = dist.size(1), dist.size(2)
        diag = torch.ones(L, dtype=torch.float)
        mask = torch.ones(L, L, dtype=torch.float) - torch.diag_embed(diag)
        mask = mask.unsqueeze(0).cuda()
        eps = 1e-9
        dist1 = dist.unsqueeze(2).expand(-1, -1, L, -1)
        dist2 = dist.unsqueeze(1).expand(-1, L, -1, -1)
        pad_mask = torch.ones(1, 1, 1, V, dtype=torch.float)
        if pad_idx is not None:
            pad_mask[:, :, :, pad_idx] = .0
        pad_mask = pad_mask.cuda()
        kl = (dist1 * torch.log(dist1 / (dist2 + eps) + eps) * pad_mask).sum(-1)
        kl = (kl * mask).sum(-1).sum(-1) / (L * (L - 1))
        return - kl.mean()

    @staticmethod
    def entropy_restrain(dist):
        eps = 1e-9
        if len(dist.shape) == 3:
            B, L, V = dist.shape
            dist = dist.reshape(-1, V)
        else:
            B = dist.size(1)
        entropy = (dist * torch.log(dist + eps)).sum() / B
        return - entropy

