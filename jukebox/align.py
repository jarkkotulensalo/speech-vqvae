"""
Get alignment from attn values
1. run a forward pass on each hop, get attn values
2. concat for all hops
"""
import numpy as np
import torch as t
from jukebox.utils.torch_utils import assert_shape, empty_cache
from jukebox.hparams import Hyperparams
from jukebox.make_models import make_model
from jukebox.save_html import save_html
from jukebox.utils.sample_utils import get_starts
import fire

def get_alignment(x, zs, labels, prior, fp16, hps):
    level = hps.levels - 1 # Top level used
    n_ctx, n_tokens = prior.n_ctx, prior.n_tokens
    n_ctx = np.int(n_ctx)
    z = zs[level]
    bs, total_length = z.shape[0], z.shape[1]
    if total_length < n_ctx:
        padding_length = n_ctx - total_length
        z = t.cat([z, t.zeros(bs, n_ctx - total_length, dtype=z.dtype, device=z.device)], dim=1)
        total_length = z.shape[1]
    else:
        padding_length = 0

    hop_length = int(hps.hop_fraction[level]*prior.n_ctx)
    n_head = prior.prior.transformer.n_head
    alignment_head, alignment_layer = prior.alignment_head, prior.alignment_layer
    print(f"alignment_head {alignment_head}, alignment_layer {alignment_layer}")
    attn_layers = set([alignment_layer])
    alignment_hops = {}
    indices_hops = {}

    prior.cuda()
    empty_cache()
    for start in get_starts(total_length, n_ctx, hop_length):
        end = start + n_ctx

        # set y offset, sample_length and lyrics tokens
        print(f"alignment.py prior.get_y start {start} end {end}")
        y, indices_hop = prior.get_y(labels, start, get_indices=True)
        print(f"alignment.py y.shape {y.shape} indices_hop.shape {indices_hop.shape}")
        print(f"alignment.py y {y}")
        assert len(indices_hop) == bs
        for indices in indices_hop:
            assert len(indices) == n_tokens

        z_bs = t.chunk(z, bs, dim=0)
        y_bs = t.chunk(y, bs, dim=0)
        w_hops = []
        for z_i, y_i in zip(z_bs, y_bs):
            print(f"z_i {z_i.shape}")
            print(f"z_i[:,start:end] {z_i[:,start:end].shape}")
            print(f"y_i.shape {y_i.shape}")
            print(f"get_attn_weights attn_layers {attn_layers}")
            # w_hop = prior.z_forward(z_i[:,start:end], [], y_i, fp16=fp16, get_attn_weights=attn_layers)
            w_hop = prior.z_forward(z_i[:, start:end], [], y_i, fp16=fp16, get_attn_weights=attn_layers)
            # print(f"w_hop.shape {w_hop}")
            print(f"w_hop[0] {w_hop[0].shape}")

            assert len(w_hop) == 1
            print(f"w_hop[0][:, alignment_head] {w_hop[0][:, alignment_head].shape}")
            w_hops.append(w_hop[0][:, alignment_head])
            del w_hop

        print(f"len(w_hops) {len(w_hops)} w_hops {w_hops[0].shape}")
        w = t.cat(w_hops, dim=0)
        del w_hops
        assert_shape(w, (bs, n_ctx, n_tokens))
        alignment_hop = w.float().cpu().numpy()
        assert_shape(alignment_hop, (bs, n_ctx, n_tokens))
        del w

        # alignment_hop has shape (bs, n_ctx, n_tokens)
        # indices_hop is a list of len=bs, each entry of len hps.n_tokens
        indices_hops[start] = indices_hop
        alignment_hops[start] = alignment_hop
    prior.cpu()
    empty_cache()

    # Combine attn for each hop into attn for full range
    # Use indices to place them into correct place for corresponding source tokens
    alignments = []
    for item in range(bs):
        # Note each item has different length lyrics

        # full_tokens = labels['info'][item]['full_tokens']
        full_tokens = labels
        alignment = np.zeros((total_length, len(full_tokens) + 1))
        print(f"alignment {alignment.shape}")
        for start in reversed(get_starts(total_length, n_ctx, hop_length)):
            end = start + n_ctx
            alignment_hop = alignment_hops[start][item]
            indices = indices_hops[start][item].cpu()
            assert len(indices) == n_tokens
            assert alignment_hop.shape == (n_ctx, n_tokens)
            print(f"start {start} alignment_hop {alignment_hop.shape}")
            print(f"start {start} indices {indices.shape} {indices}")
            # print(f"start {start} alignment[start:end, indices] {alignment[start:end, indices].shape}")
            alignment[start:end, indices] = alignment_hop
            print(f"start {start} alignment[start:end, indices] {alignment[start:end, indices].shape}")
        alignment = alignment[:total_length - padding_length, :-1] # remove token padding, and last lyric index
        alignments.append(alignment)
        print(f"alignment {alignment}")
    print(f"len(alignments) {len(alignments)}")
    return alignments

def save_alignment(model, device, hps):
    print(hps)
    vqvae, priors = make_model(model, device, hps, levels=[-1])

    logdir = f"{hps.logdir}/level_{0}"
    data = t.load(f"{logdir}/data.pth.tar")
    if model == '1b_lyrics':
        fp16 = False
    else:
        fp16 = True
    fp16 = False

    data['alignments'] = get_alignment(data['x'], data['zs'], data['labels'][-1], priors[-1], fp16, hps)
    t.save(data, f"{logdir}/data_align.pth.tar")
    save_html(logdir, data['x'], data['zs'], data['labels'][-1], data['alignments'], hps)

def run(model, port=29500, **kwargs):
    from jukebox.utils.dist_utils import setup_dist_from_mpi
    rank, local_rank, device = setup_dist_from_mpi(port=port)
    hps = Hyperparams(**kwargs)

    with t.no_grad():
        save_alignment(model, device, hps)

if __name__ == '__main__':
    fire.Fire(run)
