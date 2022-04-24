"""
Ability to train vq-vae and prior
First try for random inputs
Then from maestros
"""
import sys
import fire
import warnings
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch as t
# import jukebox.utils.dist_adapter as dist
from jukebox.utils import dist_adapter as dist
from torch.nn.parallel import DistributedDataParallel

from jukebox.hparams import setup_hparams
from jukebox.make_models import make_vqvae, make_prior, restore_opt, save_checkpoint
from jukebox.utils.logger import init_logging
from jukebox.utils.audio_utils import audio_preprocess, audio_postprocess
from jukebox.utils.torch_utils import zero_grad, count_parameters
from jukebox.utils.dist_utils import print_once, allreduce, allgather
from jukebox.utils.ema import CPUEMA, FusedEMA, EMA
from jukebox.utils.fp16 import FP16FusedAdam, FusedAdam, LossScalar, clipped_grad_scale, backward
from jukebox.data.data_processor import DataProcessor
from jukebox.data.text_processor import TextProcessor
from jukebox.data.wav_to_tokens import discretise_dataset

from jukebox.utils.train_utils import create_codebook_dataset, _plot_sample_wav, _plot_mel_spec, _plot_codebook_sample, _plot_codebook_histogram, \
    plot_alignment_to_numpy, plot_previous_token_alignment


def prepare_aud(x, hps):
    x = audio_postprocess(x.detach().contiguous(), hps)
    return allgather(x)


def log_aud(logger, tag, x, hps):
    # print(f"log_aud x.shape {x.shape}")
    logger.add_audios(tag, prepare_aud(x, hps), hps.sr, max_len=hps.max_len, max_log=hps.max_log)
    logger.flush()


def log_labels(logger, labeller, tag, y, hps):
    y = y.cpu().numpy()
    txt = ''
    for item in range(y.shape[0]):
        description = labeller.describe_label(y[item])
        lyrics = description['lyrics']
        txt += f"{item} lyrics:{lyrics}\n"
    logger.add_text(tag, txt)
    logger.flush()


def get_ddp(model, hps):
    rank = dist.get_rank()
    local_rank = rank % 8
    ddp = DistributedDataParallel(model,
                                  device_ids=[local_rank],
                                  output_device=local_rank,
                                  broadcast_buffers=False,
                                  bucket_cap_mb=hps.bucket,
                                  find_unused_parameters=True)
    return ddp


def get_ema(model, hps):
    mu = hps.mu or (1. - (hps.bs * hps.ngpus/8.)/1000)
    ema = None
    if hps.ema and hps.train:
        if hps.cpu_ema:
            if dist.get_rank() == 0:
                print("Using CPU EMA")
            ema = CPUEMA(model.parameters(), mu=mu, freq=hps.cpu_ema_freq)
        elif hps.ema_fused:
            ema = FusedEMA(model.parameters(), mu=mu)
        else:
            ema = EMA(model.parameters(), mu=mu)
    return ema


def get_lr_scheduler(opt, hps):
    def lr_lambda(step):
        if hps.lr_use_linear_decay:
            lr_scale = hps.lr_scale * min(1.0, step / hps.lr_warmup)
            decay = max(0.0, 1.0 - max(0.0, step - hps.lr_start_linear_decay) / hps.lr_decay)
            if decay == 0.0:
                if dist.get_rank() == 0:
                    print("Reached end of training")
            return lr_scale * decay
        else:
            return hps.lr_scale * (hps.lr_gamma ** (step // hps.lr_decay)) * min(1.0, step / hps.lr_warmup)

    shd = t.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    return shd


def get_optimizer(model, hps):
    # Optimizer
    betas = (hps.beta1, hps.beta2)
    if hps.fp16_opt:
        opt = FP16FusedAdam(model.parameters(), lr=hps.lr, weight_decay=hps.weight_decay, betas=betas, eps=hps.eps)
    else:
        opt = FusedAdam(model.parameters(), lr=hps.lr, weight_decay=hps.weight_decay, betas=betas, eps=hps.eps)

    """
    UserWarning: Detected call of `lr_scheduler.step()` before `optimizer.step()`. 
    In PyTorch 1.1.0 and later, you should call them in the opposite order: `optimizer.step()` 
    before `lr_scheduler.step()`. 
    Failure to do this will result in PyTorch skipping the first value of the learning rate schedule.
    """
    # lr scheduler
    shd = get_lr_scheduler(opt, hps)

    restore_path = hps.restore_prior if hps.prior else hps.restore_vqvae
    restore_opt(opt, shd, restore_path)

    # fp16 dynamic loss scaler
    scalar = None
    if hps.fp16:
        rank = dist.get_rank()
        local_rank = rank % 8
        scalar = LossScalar(hps.fp16_loss_scale, scale_factor=2 ** (1./hps.fp16_scale_window))
        if local_rank == 0:
            print(scalar.__dict__)

    zero_grad(model)
    return opt, shd, scalar


def log_inputs(orig_model, logger, x_in, y, x_out, hps, tag="train"):
    # print(f"Logging {tag} inputs/outputs")
    log_aud(logger, f'{tag}_x_original', x_in, hps)
    log_aud(logger, f'{tag}_x_reconstruction', x_out, hps)
    bs = x_in.shape[0]
    if hps.prior:
        if hps.labels:
            log_labels(logger, orig_model.labeller, f'{tag}_y_in', allgather(y.cuda()), hps)
    #else:
    #    zs_in = orig_model.encode(x_in, start_level=0, bs_chunks=bs)
    #    x_ds = [orig_model.decode(zs_in[level:], start_level=level, bs_chunks=bs) for level in range(0, hps.levels)]
    #    for i in range(len(x_ds)):
    #        log_aud(logger, f'{tag}_x_ds_start_{i}', x_ds[i], hps)
    logger.flush()

def log_attention_alignment(orig_model, z, y, hps, info=""):
    """
    Plot attention visualisation of the current autoregressive Transformer model based on predefined
    attention layer in hps
    Args:
        orig_model: Pytorch model
        z: VQVAE codes z
        y: string, sentence
        hps: dict
    Returns: None
    """
    # Lyric attention visualisation
    # layers_text = [18,28,38,48,58,68]
    layers_text = [18,28]
    for layer in layers_text:
        attn_layers = set([layer])
        w_hop = orig_model.z_forward(z, [], y, fp16=hps.fp16, get_attn_weights=attn_layers, hps=hps)
        # alignment = w_hop[0][:, hps.alignment_head]  # torch.Size([32, 3584, 128])
        # Choose which attention head to plot
        for head in range(hps.heads):
            alignment = w_hop[0][:, head]  # choose first sample from batch
            #print(f"layer {layer} w_hop[0][:, :] {w_hop[0][:, :].shape}, alignment {alignment.shape}")
            # print(f"alignment {alignment.shape}")
            alignment = alignment.cpu().numpy().astype('float64')
            plot_alignment_to_numpy(alignment=alignment, layer=layer, head=head, hps=hps, info=info)


    # plot attention to previous tokens
    layers_all = list(np.arange(0, int(hps.prior_depth), 1))
    layers_prev = set(layers_all).symmetric_difference(layers_text)
    #print(f"layers_prev {layers_prev}")

    for layer in layers_prev:
        attn_layers = set([layer])
        w_hop = orig_model.z_forward(z, [], y, fp16=hps.fp16, get_attn_weights=attn_layers, hps=hps)
        # alignment = w_hop[0][:, hps.alignment_head]  # torch.Size([32, 3584, 128])
        # print(f"layer {layer} w_hop {w_hop.shape}")

        # Choose which attention head to plot
        for head in range(hps.heads):
            alignment = w_hop[0][:, head]  # choose first sample from batch
            #print(f"layer {layer} w_hop[0][:, :] {w_hop[0][:, :].shape}, alignment {alignment.shape}")
            alignment = alignment.cpu().numpy().astype('float64')
            if layer in layers_text:
                plot_alignment_to_numpy(alignment=alignment, layer=layer, head=head, hps=hps, info=info)
            else:
                plot_previous_token_alignment(alignment=alignment, layer=layer, head=head, hps=hps, info=info)


def sample_prior(orig_model, logger, x_in, y, waveform, hps, info=""):

    # hps.bs_sample = 2
    # print(f"Sampling {hps.bs_sample} samples")  // 16
    # (f"x_in {x_in.shape}")  // torch.Size([16, 7166])
    x_in = x_in[:hps.bs_sample, :]
    # print(f"sample_prior x_in {x_in.shape}")  // torch.Size([16, 7166])
    bs = x_in.shape[0]

    #print(f"zs_in[level:] {zs_in[0:].shape}")
    # print(f"hps.levels {hps.levels}")  // 1
    # x_ds = [orig_model.decode(zs_in[level:], start_level=level, bs_chunks=bs) for level in range(0, hps.levels)]
    zs_in = x_in
    z_conds = []

    x_ds = [orig_model.decode([zs_in, *z_conds])]

    if not hps.labels:
        y = None
    elif hps.level == (hps.levels - 1):
        # Topmost level labels in order
        y = y[:hps.bs_sample]  # t.ones((hps.bs_sample, 1), device=y.device, dtype=t.long) * dist.get_rank()
    else:
        # Other levels keep labels to match x_cond
        y = y[:hps.bs_sample]

    # Modified sampling
    z = orig_model.sample(hps.bs_sample, z_conds=None, y=y, fp16=False, temp=1.0)
    # print(f"Sampled z {z.shape}")  // torch.Size([16, 7166])

    log_attention_alignment(orig_model, z, y, hps, info=f"{info}_sample")
    log_attention_alignment(orig_model, zs_in, y, hps, info=f"{info}_orig")

    z_conds = []
    x_sample = orig_model.decode([z, *z_conds], bs_chunks=bs)
    # print(f"Sampled x_sample {x_sample.shape}")  // torch.Size([16, 1, 229312])
    log_str = f'{info}_sample_x_T1'
    log_aud(logger, log_str, x_sample, hps)
    if hps.prior and hps.labels:
        log_labels(logger, orig_model.labeller, log_str, allgather(y.cuda()), hps)

    # print(f"{info} sample_prior waveform {waveform.shape}")
    # Recons
    for i in range(len(x_ds)):
        # print(f"log reconst - x_ds[i] {x_ds[i].shape}")
        log_aud(logger, f'{info}_recons_{i}', x_ds[i], hps)

        # Original sample
        # print(f"log original - waveform {waveform.shape}")
        log_aud(logger, f'{info}_orig_{i}', waveform, hps)

    logger.flush()


def evaluate(model, orig_model, logger, metrics, data_processor, hps):
    model.eval()
    orig_model.eval()

    # Use train to keep dropout at prior forward pass
    if hps.prior:
        orig_model.prior.x_emb_dropout.train()

    if hps.prior:
        _print_keys = dict(l="loss", bpd="bpd")
    else:
        _print_keys = dict(l="loss", rl="recons_loss", sl="spectral_loss")

    device = t.device("cuda" if t.cuda.is_available() else "cpu")

    embeddings_level = []
    embeddings_level2 = []
    with t.no_grad():
        for i, batch in logger.get_range(data_processor.test_loader):
            if isinstance(batch, (tuple, list)):
                # x, y = x
                if hps.prior:
                    waveform, normalized_transcript, z = batch
                    # print(f"waveform {waveform.shape}, z {z.shape}")
                    x = z
                    x = x.squeeze(1)  # torch.Size([batch_size, 7166])
                    waveform = waveform.to(device, non_blocking=True)
                    if i == 0:
                        print(f"Sample z {x[0,:]}")
                else:
                    waveform, normalized_transcript = batch
                    x = waveform

                # print(f"x {x.shape}, y: {normalized_transcript.shape}")
                y = normalized_transcript.squeeze(1)
                # print(f"x {x.shape}, y: {y.shape}")
            else:
                y = None
            x = x.to('cuda', non_blocking=True)
            if y is not None:
                y = y.to('cuda', non_blocking=True)

            # DO WE NEED AUDIO_PREPOCESS
            # x_in = x = audio_preprocess(x, hps)
            x_in = x

            log_input_output = (i == 0)

            if hps.prior:
                forw_kwargs = dict(y=y, fp16=hps.fp16, decode=log_input_output, hps=hps)
            else:
                forw_kwargs = dict(loss_fn=hps.loss_fn, hps=hps)

            x_out, loss, _metrics = model(x, **forw_kwargs)

            # Logging
            for key, val in _metrics.items():
                _metrics[key] = val.item()
            _metrics["loss"] = loss = loss.item() # Make sure to call to free graph

            # Average and log
            for key, val in _metrics.items():
                _metrics[key] = metrics.update(f"test_{key}", val, x.shape[0])

            with t.no_grad():
                if log_input_output:
                    if not hps.prior:
                        print(f"Logging test input - output")
                        log_inputs(orig_model, logger, x_in, y, x_out, hps)

                        # Plot sample waveform
                        wav_in = x_in.cpu().numpy()[0, :, :].squeeze(0)
                        wav_out = x_out.cpu().numpy()[0, :, :].squeeze(0)
                        _plot_sample_wav(wav_in, hps)
                        _plot_mel_spec(wav_in, wav_out, hps)
                        zs = orig_model.encode(x)
                        for level in range(hps.levels):
                            sample = zs[level][0, :].cpu().numpy()
                            _plot_codebook_sample(sample, wav_in, wav_out, level, hps)

                    # Plot test attention
                    # if (logger.iters % 6000) in list(range(1, 1 + hps.iters_before_update)) or finished_training:
                    if hps.prior:
                        print(f"Logging prior test alignment")
                        # print(f"First batch, first y: {y[0]}")
                        # orig_model.train()
                        sample_prior(orig_model, logger, x_in, y, waveform, hps, info='test')
                        # log_attention_alignment(orig_model, x_in, y, hps, info='test')
                        # orig_model.eval()

            logger.set_postfix(**{print_key: _metrics[key] for print_key, key in _print_keys.items()})

            # Evaluate codebook
            ###################
            # We want the final size to equal tacotron sixe of 12.5ms ~ 1764
            # Chorowski et al. (2019) used sequences of length 5120 time-domain samples (320 ms)
            if not hps.prior:
                zs = orig_model.encode(x)

                # print(f"encode zs {zs[0].shape}")  # (4 x 400) (batches x dimensions)
                for batch_num in range(zs[0].shape[0]):
                    emb_level1 = zs[0][batch_num, :]
                    embeddings_level.append(emb_level1)
                    # print(f"emb_level1 {emb_level1.shape}")  # (torch.Size([400]))
                    if hps.levels > 1:
                        emb_level2 = zs[1][batch_num, :]
                        embeddings_level2.append(emb_level2)
            ###################

    # Evaluate codebook
    ###################
    # TODO: Create list of lists to streamline code
    if not hps.prior:
        for level in range(hps.levels):
            # Reduce dimension
            if level == 0:
                codebook = np.array([t.cpu().numpy() for t in embeddings_level])  # (1310, 1600)
            else:
                codebook = np.array([t.cpu().numpy() for t in embeddings_level2])  # (1310, 400)
            _plot_codebook_histogram(codebook, level, hps)
    ###################s

    for key, val in _metrics.items():
        logger.add_scalar(f"test_{key}", metrics.avg(f"test_{key}"))

    logger.close_range()
    return {key: metrics.avg(f"test_{key}") for key in _metrics.keys()}


def train(model, orig_model, opt, shd, scalar, ema, logger, metrics, data_processor, hps, epoch):
    model.train()
    orig_model.train()
    if hps.prior:
        _print_keys = dict(l="loss", bpd="bpd", gn="gn", g_l="gen_loss", p_l="prime_loss")
    else:
        _print_keys = dict(l="loss", sl="spectral_loss", rl="recons_loss", e="entropy", u="usage", uc="used_curr", gn="gn", pn="pn", dk="dk")

    print(f"hps.save_iters {hps.save_iters}")

    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    # print(f"device is {device}")

    for i, batch in logger.get_range(data_processor.train_loader):
        # print(f"batch {batch}")
        if batch is None:
            continue
        if isinstance(batch, (tuple, list)):
            if hps.prior:
                waveform, normalized_transcript, z = batch
                # print(f"waveform {waveform.shape}, z {z.shape}")
                x = z
                x = x.squeeze(1)  # torch.Size([batch_size, 7166])
                waveform = waveform.to(device, non_blocking=True)
            else:
                waveform, normalized_transcript = batch
                x = waveform

            # print(f"train x.type {x.type()}")
            # print(f"x {x.shape}, y: {normalized_transcript.shape}")
            y = normalized_transcript.squeeze(1)  # (torch.Size([batch_size, 188]))
            #
            # print(f"x {x.shape}, y: {y.shape}")

        else:
            x = batch
            y = None

        # print(f"x {x.shape}, y: {y.shape}")  # x torch.Size([128, 1, 12800]), y: torch.Size([128, 192])

        x = x.to(device, non_blocking=True)
        # print(f"train x.to(device).type {x.type()}")
        if y is not None:
            y = y.to(device, non_blocking=True)
            # print(f"x.device {x.device} y.device {y.device}")

        x_in = x
        log_input_output = (logger.iters % hps.save_iters == 0)

        if hps.prior:
            forw_kwargs = dict(y=y, fp16=hps.fp16, decode=log_input_output, hps=hps)
        else:
            forw_kwargs = dict(loss_fn=hps.loss_fn, hps=hps)

        # Forward
        # print(f"train.py forward x {x.shape}, y {y.shape}, hps.fp16 {hps.fp16}, log_input_output {log_input_output}")
        x_out, loss, _metrics = model(x, **forw_kwargs)

        # Backward
        loss, scale, grad_norm, overflow_loss, overflow_grad = backward(loss=loss,
                                                                        params=list(model.parameters()),
                                                                        scalar=scalar,
                                                                        fp16=hps.fp16,
                                                                        logger=logger)
        # Skip step if overflow
        grad_norm = allreduce(grad_norm, op=dist.ReduceOp.MAX)
        if overflow_loss or overflow_grad or grad_norm > hps.ignore_grad_norm > 0:
            zero_grad(orig_model)
            continue

        # Step opt. Divide by scale to include clipping and fp16 scaling
        # logger.step()
        # opt.step(scale=clipped_grad_scale(grad_norm, hps.clip, scale))
        logger.step()
        opt.step(scale=clipped_grad_scale(grad_norm, hps.clip, scale))
        zero_grad(orig_model)

        ############## MODDED
        # lr = hps.lr if shd is None else shd.get_lr()[0]
        # print(f"shd.get_lr()[0] {shd.get_lr()[0]}")
        # print(f"shd.get_last_lr() {shd.get_last_lr()}")
        # print(f"shd.get_last_lr()[0] {shd.get_last_lr()[0]}")
        lr = hps.lr if shd is None else shd.get_last_lr()[0]
        if shd is not None:
            shd.step()
        if ema is not None:
            ema.step()
        # next_lr = hps.lr if shd is None else shd.get_lr()[0]
        next_lr = hps.lr if shd is None else shd.get_last_lr()[0]
        ##############

        finished_training = (next_lr == 0.0)

        # Logging
        for key, val in _metrics.items():
            _metrics[key] = val.item()
        _metrics["loss"] = loss = loss.item() * hps.iters_before_update # Make sure to call to free graph
        _metrics["gn"] = grad_norm
        _metrics["lr"] = lr
        _metrics["lg_loss_scale"] = np.log2(scale)

        # Average and log
        for key, val in _metrics.items():
            _metrics[key] = metrics.update(key, val, x.shape[0])
            if logger.iters % hps.log_steps == 0:
                logger.add_scalar(key, _metrics[key])

        # Save checkpoint
        with t.no_grad():
            if hps.save and (logger.iters % hps.save_iters == 1 or finished_training):
                print(f"Save checkpoint at logger.iters {logger.iters}")
                if ema is not None:
                    ema.swap()
                orig_model.eval()
                name = 'latest' if hps.prior else f'step_{logger.iters}'
                if dist.get_rank() % 8 == 0:
                    save_checkpoint(logger, name, orig_model, opt, dict(step=logger.iters), hps)
                orig_model.train()
                if ema is not None:
                    ema.swap()

        # Sample and plot attention alignment plots
        # print(f"finished_training = (next_lr == 0.0) <> {next_lr} ")
        # print(f"list(range(1, 1 + hps.iters_before_update)) {list(range(1, 1 + hps.iters_before_update))}")
        with t.no_grad():
            # if (logger.iters % 6000) in list(range(1, 1 + hps.iters_before_update)) or finished_training:
            # if logger.iters % hps.save_iters == 1:
            if log_input_output:
                print(f"Sampling at logger.iters {logger.iters}")
                if hps.prior:
                    if ema is not None:
                        ema.swap()
                    orig_model.eval()

                    orig_model.prior.x_emb_dropout.train()
                    sample_prior(orig_model, logger, x_in, y, waveform, hps, info='train')
                    # log_attention_alignment(orig_model, x_in, y, hps, info='train')
                    orig_model.train()
                    if ema is not None:
                        ema.swap()
                    # print(f"removing sampling")

        # Input/Output
        with t.no_grad():
            if log_input_output:
                if not hps.prior:
                    print(f"Save inputs at logger.iters {logger.iters}")
                    log_inputs(orig_model, logger, x_in, y, x_out, hps)

        logger.set_postfix(**{print_key: _metrics[key] for print_key, key in _print_keys.items()})
        if finished_training:
            dist.barrier()
            exit()

    logger.close_range()

    #print(f"max_tokens in all dataset sentences is {data_processor.max_tokens}")
    return {key: metrics.avg(key) for key in _metrics.keys()}


def run(hps="vqvae", port=29500, **kwargs):
    from jukebox.utils.dist_utils import setup_dist_from_mpi
    rank, local_rank, device = setup_dist_from_mpi(port=port)
    hps = setup_hparams(hps, kwargs)
    hps.ngpus = dist.get_world_size()
    hps.argv = " ".join(sys.argv)
    hps.bs_sample = hps.nworkers = hps.bs

    print(f"hps.prior {hps.prior}, hps.train {hps.train}, hps.test {hps.test}, hps.discretise {hps.discretise}")
    print(f"hps: l_bins {hps.l_bins}")

    # Setup models
    vqvae = make_vqvae(hps, device)
    print_once(f"Parameters VQVAE:{count_parameters(vqvae)}")

    if hps.prior:
        segment = False
        if hps.discretise:
            print(f"Starting discretise dataset")
            discretise_dataset(vqvae=vqvae,
                               in_dirpath=f'jukebox/datasets/LJSpeech-1.1/wavs',
                               out_dirpath=f'jukebox/datasets/LJSpeech-1.1/tokens',
                               device=device)
            print(f"Discretise dataset ready")

        prior = make_prior(hps, vqvae, device)
        print_once(f"Parameters Prior:{count_parameters(prior)}")
        model = prior
        # print(f"prior model is {model}")
    else:
        model = vqvae
        segment = False
        # print(f"vqvae model set.")

    # Setup dataset
    data_processor = DataProcessor(hps, segment=segment)

    # Setup opt, ema and distributed_model.
    opt, shd, scalar = get_optimizer(model, hps)
    print(f"opt is {opt}, shd {shd}, scalar {scalar}")
    ema = get_ema(model, hps)
    print(f"ema is {ema}")
    # print(f"model is {model}")
    distributed_model = get_ddp(model, hps)

    logger, metrics = init_logging(hps, local_rank, rank)
    logger.iters = model.step

    # Run training, eval, sample
    # hps.epochs = 50
    for epoch in range(hps.curr_epoch, hps.epochs):
        print(f"Running {epoch + 1}/{hps.epochs} epochs")
        metrics.reset()
        data_processor.set_epoch(epoch)

        if hps.train:
            print(f"Starting training")
            train_metrics = train(distributed_model, model, opt, shd, scalar, ema, logger, metrics, data_processor, hps, epoch)
            train_metrics['epoch'] = epoch
            if rank == 0:
                print('Train', ' '.join([f'{key}: {val:0.4f}' for key,val in train_metrics.items()]))
            dist.barrier()

        if hps.test:
            print(f"Starting testing")
            if ema:
                ema.swap()
            test_metrics = evaluate(distributed_model, model, logger, metrics, data_processor, hps)
            test_metrics['epoch'] = epoch
            if rank == 0:
                print('Ema', ' '.join([f'{key}: {val:0.4f}' for key,val in test_metrics.items()]))
            dist.barrier()
            if ema:
                ema.swap()

        dist.barrier()
    print(f"Training ready.")


if __name__ == '__main__':
    fire.Fire(run)
