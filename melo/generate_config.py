#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
既存のlistファイルからconfig.jsonを生成するスクリプト
"""

import json
import os
from collections import defaultdict
from tqdm import tqdm
import click
from text.symbols import symbols, num_languages, num_tones

@click.command()
@click.argument("list_dir", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--cleaned-path", default=None, help="metadata.list.cleanedのパス")
@click.option("--train-path", default=None, help="train.listのパス")
@click.option("--val-path", default=None, help="val.listのパス")
@click.option("--output", default=None, help="出力するconfig.jsonのパス")
@click.option("--template", default=None, help="テンプレートとして使用するconfig.jsonのパス")
def main(list_dir, cleaned_path, train_path, val_path, output, template):
    """
    既存のlistファイルからconfig.jsonを生成します。
    """
    # デフォルトパスの設定
    if cleaned_path is None:
        cleaned_path = os.path.join(list_dir, "metadata.list.cleaned")
    if train_path is None:
        train_path = os.path.join(list_dir, "train.list")
    if val_path is None:
        val_path = os.path.join(list_dir, "val.list")
    if output is None:
        output = os.path.join(list_dir, "config.json")
    if template is None:
        template = os.path.join(os.path.dirname(os.path.dirname(list_dir)), "data/test_line_distilbert/config.json")
        if not os.path.exists(template):
            template = None
    
    # ファイルの存在確認
    for path, name in [(cleaned_path, "metadata.list.cleaned"), (train_path, "train.list"), (val_path, "val.list")]:
        if not os.path.exists(path):
            print(f"エラー: {name}が見つかりません: {path}")
            return
    
    # スピーカー情報の処理
    print("スピーカー情報を処理中...")
    spk_utt_map = defaultdict(list)
    spk_id_map = {}
    current_sid = 0

    with open(cleaned_path, encoding="utf-8") as f:
        for line in tqdm(f.readlines(), desc="スピーカー情報読み込み中"):
            try:
                parts = line.strip().split("|")
                if len(parts) >= 7:
                    utt, spk, language, text, phones, tones, word2ph = parts[:7]
                    spk_utt_map[spk].append(line)
                    if spk not in spk_id_map:
                        spk_id_map[spk] = current_sid
                        current_sid += 1
                else:
                    print(f"Warning: 不正な形式の行をスキップ: {line.strip()}")
            except Exception as e:
                print(f"Error processing line: {line.strip()}\nError: {str(e)}")
                continue
    
    print(f"スピーカー数: {len(spk_id_map)}")
    
    # トレーニングデータと検証データの行数を確認
    train_lines = 0
    val_lines = 0
    
    with open(train_path, encoding="utf-8") as f:
        train_lines = len(f.readlines())
    
    with open(val_path, encoding="utf-8") as f:
        val_lines = len(f.readlines())
    
    print(f"トレーニングデータ: {train_lines}行")
    print(f"検証データ: {val_lines}行")
    
    # config.jsonの生成
    if template and os.path.exists(template):
        print(f"テンプレートを使用: {template}")
        with open(template, encoding="utf-8") as f:
            config = json.load(f)
    else:
        print("デフォルト設定を使用")
        # デフォルト設定
        config = {
            "train": {
                "log_interval": 200,
                "eval_interval": 1000,
                "seed": 52,
                "epochs": 10000,
                "learning_rate": 0.0003,
                "betas": [0.8, 0.99],
                "eps": 1e-09,
                "batch_size": 6,
                "fp16_run": False,
                "lr_decay": 0.999875,
                "segment_size": 16384,
                "init_lr_ratio": 1,
                "warmup_epochs": 0,
                "c_mel": 45,
                "c_kl": 1.0,
                "skip_optimizer": True
            },
            "data": {
                "max_wav_value": 32768.0,
                "sampling_rate": 44100,
                "filter_length": 2048,
                "hop_length": 512,
                "win_length": 2048,
                "n_mel_channels": 128,
                "mel_fmin": 0.0,
                "mel_fmax": None,
                "add_blank": True,
                "cleaned_text": True,
                "spk2id": {},
                "n_speakers": 0,
                "training_files": "",
                "validation_files": ""
            },
            "model": {
                "use_spk_conditioned_encoder": True,
                "use_noise_scaled_mas": True,
                "use_mel_posterior_encoder": False,
                "use_duration_discriminator": True,
                "inter_channels": 192,
                "hidden_channels": 192,
                "filter_channels": 768,
                "n_heads": 2,
                "n_layers": 6,
                "n_layers_trans_flow": 3,
                "kernel_size": 3,
                "p_dropout": 0.1,
                "resblock": "1",
                "resblock_kernel_sizes": [3, 7, 11],
                "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                "upsample_rates": [8, 8, 2, 2, 2],
                "upsample_initial_channel": 512,
                "upsample_kernel_sizes": [16, 16, 8, 2, 2],
                "n_layers_q": 3,
                "use_spectral_norm": False,
                "gin_channels": 256
            },
            "num_languages": 0,
            "num_tones": 0,
            "symbols": []
        }
    
    # 設定を更新
    config["data"]["spk2id"] = spk_id_map
    config["data"]["training_files"] = train_path
    config["data"]["validation_files"] = val_path
    config["data"]["n_speakers"] = len(spk_id_map)
    config["num_languages"] = num_languages
    config["num_tones"] = num_tones
    config["symbols"] = symbols
    
    # 設定を保存
    with open(output, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"設定ファイルを保存しました: {output}")

if __name__ == "__main__":
    main() 