# train.py

# マルチプロセス設定
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

import os
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import logging

logging.getLogger("numba").setLevel(logging.WARNING)
import commons
import utils
from data_utils import (
    TextAudioSpeakerLoader,
    TextAudioSpeakerCollate,
    DistributedBucketSampler,
)
from models import (
    SynthesizerTrn,
    MultiPeriodDiscriminator,
    DurationDiscriminator,
)
from losses import generator_loss, discriminator_loss, feature_loss, kl_loss
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from text.symbols import symbols
from melo.download_utils import load_pretrain_model

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = (
    True  # If encontered training problem,please try to disable TF32.
)
torch.set_float32_matmul_precision("medium")


torch.backends.cudnn.benchmark = True
torch.backends.cuda.sdp_kernel("flash")
torch.backends.cuda.enable_flash_sdp(True)
# torch.backends.cuda.enable_mem_efficient_sdp(
#     True
# )  # Not available if torch version is lower than 2.0
torch.backends.cuda.enable_math_sdp(True)
global_step = 0


def run():
    hps = utils.get_hparams()
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(
        backend="gloo",
        init_method="env://",  # Due to some training problem,we proposed to use gloo instead of nccl.
        rank=local_rank,
    )  # Use torchrun instead of mp.spawn
    rank = dist.get_rank()
    n_gpus = dist.get_world_size()
    
    torch.manual_seed(hps.train.seed)
    torch.cuda.set_device(rank)
    global global_step
    if rank == 0:
        logger = utils.get_logger(hps.model_dir)
        logger.info(hps)
        utils.check_git_hash(hps.model_dir)
        writer = SummaryWriter(log_dir=hps.model_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))
    train_dataset = TextAudioSpeakerLoader(hps.data.training_files, hps.data)
    train_sampler = DistributedBucketSampler(
        train_dataset,
        hps.train.batch_size,
        [32, 300, 400, 500, 600, 700, 800, 900, 1000],
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True,
    )
    collate_fn = TextAudioSpeakerCollate()
    train_loader = DataLoader(
        train_dataset,
        num_workers=8,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
        batch_sampler=train_sampler,
        persistent_workers=True,
        prefetch_factor=4,
    )  # DataLoader config could be adjusted.
    if rank == 0:
        eval_dataset = TextAudioSpeakerLoader(hps.data.validation_files, hps.data)
        eval_loader = DataLoader(
            eval_dataset,
            num_workers=0,
            shuffle=False,
            batch_size=1,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )
    if (
        "use_noise_scaled_mas" in hps.model.keys()
        and hps.model.use_noise_scaled_mas is True
    ):
        print("Using noise scaled MAS for VITS2")
        mas_noise_scale_initial = 0.01
        noise_scale_delta = 2e-6
    else:
        print("Using normal MAS for VITS1")
        mas_noise_scale_initial = 0.0
        noise_scale_delta = 0.0
    if (
        "use_duration_discriminator" in hps.model.keys()
        and hps.model.use_duration_discriminator is True
    ):
        print("Using duration discriminator for VITS2")
        net_dur_disc = DurationDiscriminator(
            hps.model.hidden_channels,
            hps.model.hidden_channels,
            3,
            0.1,
            gin_channels=hps.model.gin_channels if hps.data.n_speakers != 0 else 0,
        ).cuda(rank)
    if (
        "use_spk_conditioned_encoder" in hps.model.keys()
        and hps.model.use_spk_conditioned_encoder is True
    ):
        if hps.data.n_speakers == 0:
            raise ValueError(
                "n_speakers must be > 0 when using spk conditioned encoder to train multi-speaker model"
            )
    else:
        print("Using normal encoder for VITS1")

    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        mas_noise_scale_initial=mas_noise_scale_initial,
        noise_scale_delta=noise_scale_delta,
        **hps.model,
    ).cuda(rank)

    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda(rank)
    optim_g = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, net_g.parameters()),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    if net_dur_disc is not None:
        optim_dur_disc = torch.optim.AdamW(
            net_dur_disc.parameters(),
            hps.train.learning_rate,
            betas=hps.train.betas,
            eps=hps.train.eps,
        )
    else:
        optim_dur_disc = None
    net_g = DDP(net_g, device_ids=[rank], find_unused_parameters=True)
    net_d = DDP(net_d, device_ids=[rank], find_unused_parameters=True)
    
    pretrain_G, pretrain_D, pretrain_dur = load_pretrain_model()
    hps.pretrain_G = hps.pretrain_G or None
    hps.pretrain_D = hps.pretrain_D or None
    hps.pretrain_dur = hps.pretrain_dur or None

    if hps.pretrain_G:
        utils.load_checkpoint(
                hps.pretrain_G,
                net_g,
                None,
                skip_optimizer=True
            )
    if hps.pretrain_D:
        utils.load_checkpoint(
                hps.pretrain_D,
                net_d,
                None,
                skip_optimizer=True
            )


    if net_dur_disc is not None:
        net_dur_disc = DDP(net_dur_disc, device_ids=[rank], find_unused_parameters=True)
        if hps.pretrain_dur:
            utils.load_checkpoint(
                    hps.pretrain_dur,
                    net_dur_disc,
                    None,
                    skip_optimizer=True
                )
                
    try:
        if net_dur_disc is not None:
            _, _, dur_resume_lr, epoch_str = utils.load_checkpoint(
                utils.latest_checkpoint_path(hps.model_dir, "DUR_*.pth"),
                net_dur_disc,
                optim_dur_disc,
                skip_optimizer=hps.train.skip_optimizer
                if "skip_optimizer" in hps.train
                else True,
            )
            _, optim_g, g_resume_lr, epoch_str = utils.load_checkpoint(
                utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"),
                net_g,
                optim_g,
                skip_optimizer=hps.train.skip_optimizer
                if "skip_optimizer" in hps.train
                else True,
            )
            _, optim_d, d_resume_lr, epoch_str = utils.load_checkpoint(
                utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"),
                net_d,
                optim_d,
                skip_optimizer=hps.train.skip_optimizer
                if "skip_optimizer" in hps.train
                else True,
            )
            if not optim_g.param_groups[0].get("initial_lr"):
                optim_g.param_groups[0]["initial_lr"] = g_resume_lr
            if not optim_d.param_groups[0].get("initial_lr"):
                optim_d.param_groups[0]["initial_lr"] = d_resume_lr
            if not optim_dur_disc.param_groups[0].get("initial_lr"):
                optim_dur_disc.param_groups[0]["initial_lr"] = dur_resume_lr

        epoch_str = max(epoch_str, 1)
        global_step = (epoch_str - 1) * len(train_loader)
    except Exception as e:
        print(e)
        epoch_str = 1
        global_step = 0

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )
    if net_dur_disc is not None:
        scheduler_dur_disc = torch.optim.lr_scheduler.ExponentialLR(
            optim_dur_disc, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
        )
    else:
        scheduler_dur_disc = None
    scaler = GradScaler(enabled=hps.train.fp16_run)

    for epoch in range(epoch_str, hps.train.epochs + 1):
        try:
            if rank == 0:
                train_and_evaluate(
                    rank,
                    epoch,
                    hps,
                    [net_g, net_d, net_dur_disc],
                    [optim_g, optim_d, optim_dur_disc],
                    [scheduler_g, scheduler_d, scheduler_dur_disc],
                    scaler,
                    [train_loader, eval_loader],
                    logger,
                    [writer, writer_eval],
                )
            else:
                train_and_evaluate(
                    rank,
                    epoch,
                    hps,
                    [net_g, net_d, net_dur_disc],
                    [optim_g, optim_d, optim_dur_disc],
                    [scheduler_g, scheduler_d, scheduler_dur_disc],
                    scaler,
                    [train_loader, None],
                    None,
                    None,
                )
        except Exception as e:
            print(e)
            torch.cuda.empty_cache()
        scheduler_g.step()
        scheduler_d.step()
        if net_dur_disc is not None:
            scheduler_dur_disc.step()


def train_and_evaluate(rank, current_epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers):
    net_g, net_d, net_dur_disc = nets
    optim_g, optim_d, optim_dur_disc = optims
    scheduler_g, scheduler_d, scheduler_dur_disc = schedulers
    train_loader, eval_loader = loaders
    if writers is not None:
        writer, writer_eval = writers

    train_loader.batch_sampler.set_epoch(current_epoch)
    global global_step

    net_g.train()
    net_d.train()
    for batch_idx, batch in enumerate(train_loader):
        try:
            # バッチデータの形状を確認
            print("Batch shapes:")
            for i, item in enumerate(batch):
                if isinstance(item, torch.Tensor):
                    print(f"Item {i}: {item.shape}")
                else:
                    print(f"Item {i}: {type(item)}")
            
            # モデルに渡す前にデータの形状を確認
            x, x_lengths, spec, spec_lengths, wav, wav_lengths, sid, tone, language, bert, ja_bert = batch
            print("\nModel input shapes:")
            print(f"x: {x.shape}")
            print(f"x_lengths: {x_lengths.shape}")
            print(f"spec: {spec.shape}")
            print(f"spec_lengths: {spec_lengths.shape}")
            print(f"sid: {sid.shape}")
            print(f"tone: {tone.shape}")
            print(f"language: {language.shape}")
            print(f"bert: {bert.shape}")
            print(f"ja_bert: {ja_bert.shape}")

            x, x_lengths = x.cuda(rank, non_blocking=True), x_lengths.cuda(rank, non_blocking=True)
            spec, spec_lengths = spec.cuda(rank, non_blocking=True), spec_lengths.cuda(rank, non_blocking=True)
            wav, wav_lengths = wav.cuda(rank, non_blocking=True), wav_lengths.cuda(rank, non_blocking=True)
            sid = sid.cuda(rank, non_blocking=True)
            tone = tone.cuda(rank, non_blocking=True)
            language = language.cuda(rank, non_blocking=True)
            bert = bert.cuda(rank, non_blocking=True)
            ja_bert = ja_bert.cuda(rank, non_blocking=True)

            with autocast(False):
                y_hat, l_length, attn, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q), (x, logw, logw_) = net_g(
                    x, x_lengths, spec, spec_lengths, sid, tone, language, bert, ja_bert
                )
                mel = spec_to_mel_torch(
                    spec,
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax,
                )
                y_mel = commons.slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
                y_hat_mel = mel_spectrogram_torch(
                    y_hat.squeeze(1).float(),
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.hop_length,
                    hps.data.win_length,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax,
                )

                y = commons.slice_segments(wav, ids_slice * hps.data.hop_length, hps.train.segment_size)  # slice

                # Discriminator
                y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(wav, y_hat.detach())
                with autocast(False):
                    loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
                    loss_disc_all = loss_disc
                    if net_dur_disc is not None:
                        y_dur_hat_r, y_dur_hat_g = net_dur_disc(wav, y_hat.detach(), sid)
                        loss_dur_disc, losses_dur_disc_r, losses_dur_disc_g = discriminator_loss(y_dur_hat_r, y_dur_hat_g)
                        loss_disc_all += loss_dur_disc
                    loss_disc_all = loss_disc_all / hps.train.accumulation_steps
                scaler.scale(loss_disc_all).backward()
                if global_step % hps.train.accumulation_steps == 0:
                    scaler.unscale_(optim_d)
                    grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
                    scaler.step(optim_d)
                    scaler.update()
                    optim_d.zero_grad()
                    if net_dur_disc is not None:
                        scaler.unscale_(optim_dur_disc)
                        grad_norm_dur_disc = commons.clip_grad_value_(net_dur_disc.parameters(), None)
                        scaler.step(optim_dur_disc)
                        scaler.update()
                        optim_dur_disc.zero_grad()

                # Generator
                with autocast(False):
                    loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                    loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl

                    loss_fm = feature_loss(fmap_r, fmap_g)
                    loss_gen, losses_gen = generator_loss(y_d_hat_g)
                    loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl + l_length
                    loss_gen_all = loss_gen_all / hps.train.accumulation_steps
                scaler.scale(loss_gen_all).backward()
                if global_step % hps.train.accumulation_steps == 0:
                    scaler.unscale_(optim_g)
                    grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
                    scaler.step(optim_g)
                    scaler.update()
                    optim_g.zero_grad()

                if rank == 0:
                    if global_step % hps.train.log_interval == 0:
                        lr = optim_g.param_groups[0]["lr"]
                        logger.info(
                            "Train Epoch: {} [{}/{} ({:.0f}%)]\tLR: {:.6f}".format(
                                current_epoch,
                                batch_idx * len(x),
                                len(train_loader.dataset),
                                100.0 * batch_idx / len(train_loader),
                                lr,
                            )
                        )
                        logger.info(
                            f"Losses: {loss_gen_all.item():.4f} {loss_disc_all.item():.4f}"
                        )

                global_step += 1

        except Exception as e:
            print(f"Error in batch processing: {str(e)}")
            print("Batch contents:")
            for i, item in enumerate(batch):
                print(f"Item {i}: {type(item)}")
            raise

    if rank == 0:
        if current_epoch % hps.train.save_every_n_epoch == 0:
            utils.save_checkpoint(
                net_g,
                optim_g,
                hps.train.learning_rate,
                current_epoch,
                os.path.join(hps.model_dir, f"G_{global_step}.pth"),
            )
            utils.save_checkpoint(
                net_d,
                optim_d,
                hps.train.learning_rate,
                current_epoch,
                os.path.join(hps.model_dir, f"D_{global_step}.pth"),
            )
            if net_dur_disc is not None:
                utils.save_checkpoint(
                    net_dur_disc,
                    optim_dur_disc,
                    hps.train.learning_rate,
                    current_epoch,
                    os.path.join(hps.model_dir, f"DUR_{global_step}.pth"),
                )

    global epoch
    epoch += 1


def evaluate(hps, generator, eval_loader, writer_eval):
    generator.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_loader):
            try:
                x, x_lengths, spec, spec_lengths, wav, wav_lengths, sid, tone, language, bert, ja_bert = batch
                x, x_lengths = x.cuda(0), x_lengths.cuda(0)
                spec, spec_lengths = spec.cuda(0), spec_lengths.cuda(0)
                wav, wav_lengths = wav.cuda(0), wav_lengths.cuda(0)
                sid = sid.cuda(0)
                tone = tone.cuda(0)
                language = language.cuda(0)
                bert = bert.cuda(0)
                ja_bert = ja_bert.cuda(0)

                # remove else
                x = x[0, : x_lengths[0]].unsqueeze(0)
                x_lengths = x_lengths[0].unsqueeze(0)
                spec = spec[0, : spec_lengths[0]].unsqueeze(0)
                spec_lengths = spec_lengths[0].unsqueeze(0)
                wav = wav[0, : wav_lengths[0]].unsqueeze(0)
                wav_lengths = wav_lengths[0].unsqueeze(0)
                sid = sid[0].unsqueeze(0)
                tone = tone[0, : x_lengths[0]].unsqueeze(0)
                language = language[0, : x_lengths[0]].unsqueeze(0)
                bert = bert[0, :, : x_lengths[0]].unsqueeze(0)
                ja_bert = ja_bert[0, :, : x_lengths[0]].unsqueeze(0)

                y_hat, attn, mask, *_ = generator.module.infer(
                    x, x_lengths, sid, tone, language, bert, ja_bert, max_len=1000
                )
                y_hat_lengths = mask.sum([1, 2]).long() * hps.data.hop_length

                mel = spec_to_mel_torch(
                    spec,
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax,
                )
                y_hat_mel = mel_spectrogram_torch(
                    y_hat.squeeze(1).float(),
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.hop_length,
                    hps.data.win_length,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax,
                )
                image_dict = {
                    "gen/mel": utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy())
                }
                audio_dict = {"gen/audio": y_hat[0, :, : y_hat_lengths[0]].cpu().numpy()}
                if global_step == 0:
                    image_dict.update(
                        {"gt/mel": utils.plot_spectrogram_to_numpy(mel[0].cpu().numpy())}
                    )
                    audio_dict.update({"gt/audio": wav[0, :, : wav_lengths[0]].cpu().numpy()})
                utils.summarize(
                    writer=writer_eval,
                    global_step=global_step,
                    images=image_dict,
                    audios=audio_dict,
                    audio_sampling_rate=hps.data.sampling_rate,
                )
            except Exception as e:
                print(f"Error in evaluation batch {batch_idx}: {str(e)}")
                print("Batch contents:")
                for i, item in enumerate(batch):
                    print(f"Item {i}: {type(item)}")
                raise

    generator.train()


if __name__ == "__main__":
    run()
