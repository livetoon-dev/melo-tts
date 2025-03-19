# preprocess_text.py

import json
import warnings
import os
from collections import defaultdict
import torch
import multiprocessing as mp
from tqdm import tqdm
import click
from text.cleaner import clean_text_bert, get_bert_model, get_bert_tokenizer
from text.symbols import symbols, num_languages, num_tones
from transformers import AutoTokenizer, AutoModel
import sentencepiece as spm
import cProfile
import pstats
from concurrent.futures import ProcessPoolExecutor, as_completed
from random import shuffle
import gc
import time
import hashlib
import shutil

# 警告抑制
warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'true'  # 並列処理を有効化

# BERTモデルのキャッシュ
bert_model_cache = {}
bert_tokenizer_cache = {}

# GPUデバイスのキャッシュ
device_cache = {}

def get_bert_model(device: str) -> AutoModel:
    """BERTモデルを取得（キャッシュ付き）"""
    if device not in bert_model_cache:
        try:
            bert_model_cache[device] = AutoModel.from_pretrained(
                "line-corporation/line-distilbert-base-japanese",
                trust_remote_code=True
            ).to(device)
        except Exception as e:
            print(f"BERTモデルの取得に失敗: {e}")
            raise
    return bert_model_cache[device]

def get_bert_tokenizer() -> AutoTokenizer:
    """BERTトークナイザーを取得（キャッシュ付き）"""
    if 'tokenizer' not in bert_tokenizer_cache:
        try:
            bert_tokenizer_cache['tokenizer'] = AutoTokenizer.from_pretrained(
                "line-corporation/line-distilbert-base-japanese",
                trust_remote_code=True
            )
        except Exception as e:
            print(f"BERTトークナイザーの取得に失敗: {e}")
            raise
    return bert_tokenizer_cache['tokenizer']

def get_available_gpus() -> list[str]:
    """利用可能なGPUデバイスのリストを取得"""
    if not torch.cuda.is_available():
        return ["cpu"]
    return [f"cuda:{i}" for i in range(torch.cuda.device_count())]

def normalize_filepath(path: str) -> str:
    """ファイルパスを正規化し、安全な形式に変換"""
    # パスを分解
    dir_path = os.path.dirname(path)
    file_name = os.path.basename(path)
    
    # ファイル名をハッシュ化（拡張子は保持）
    name, ext = os.path.splitext(file_name)
    name_hash = hashlib.md5(name.encode()).hexdigest()[:12]  # 12文字のハッシュを使用
    safe_name = f"{name_hash}{ext}"
    
    return os.path.join(dir_path, safe_name)

def process_lines_batch(lines: list[str], device: str = "cuda:0", batch_size: int = 16) -> list[str]:
    """
    複数の行をバッチ処理する関数
    """
    results = []
    
    # 事前にBERTモデルをロード
    print(f"デバイス {device} でBERTモデルを初期化")
    try:
        get_bert_model(device)
        get_bert_tokenizer()
    except Exception as e:
        print(f"BERTモデルの初期化に失敗: {e}")
        raise
    
    # バッチに分割して処理
    for i in tqdm(range(0, len(lines), batch_size), desc="テキスト処理中", unit="バッチ"):
        batch = lines[i:i+batch_size]
        batch_results = []
        
        # バッチ内のテキストを処理
        for line in batch:
            try:
                utt, spk, language, text = line.strip().split("|")
                
                # ファイルパスを正規化
                original_wav_path = utt
                normalized_wav_path = normalize_filepath(original_wav_path)
                
                # 元のWAVファイルを新しい場所にコピー
                if os.path.exists(original_wav_path):
                    os.makedirs(os.path.dirname(normalized_wav_path), exist_ok=True)
                    shutil.copy2(original_wav_path, normalized_wav_path)
                
                # デバイスを明示的に指定
                norm_text, phones, tones, word2ph, bert = clean_text_bert(text, language, device)
                
                # BERT特徴量を保存（正規化されたパスを使用）
                bert_path = normalized_wav_path.replace(".wav", ".bert.pt")
                os.makedirs(os.path.dirname(bert_path), exist_ok=True)
                torch.save(bert, bert_path)
                
                # 処理済み行を生成（正規化されたパスを使用）
                processed_line = "{}|{}|{}|{}|{}|{}|{}\n".format(
                    normalized_wav_path,
                    spk,
                    language,
                    norm_text,
                    " ".join(phones),
                    " ".join([str(i) for i in tones]),
                    " ".join([str(i) for i in word2ph]),
                )
                batch_results.append(processed_line)
            except Exception as e:
                print(f"行の処理に失敗: {line.strip()}\nエラー: {str(e)}")
                continue
        
        # バッチ処理結果を追加
        results.extend(batch_results)
        
        # GPUメモリを定期的に解放
        if torch.cuda.is_available() and i % (batch_size * 10) == 0:
            torch.cuda.empty_cache()
            gc.collect()
    
    return results

def process_batch(lines: list[str], output_file: str, device: str = "cuda:0", batch_size: int = 16) -> None:
    """
    シングルGPUで処理を行う関数
    """
    print(f"使用するGPU: {device}, バッチサイズ: {batch_size}")
    start_time = time.time()
    
    # バッチ処理
    results = process_lines_batch(lines, device, batch_size)
    
    # 結果をファイルに書き込み
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in tqdm(results, desc="結果を保存中", unit="行"):
            f.write(result)
    
    elapsed = time.time() - start_time
    lines_per_second = len(lines) / elapsed
    print(f"処理完了: {len(lines)}行を{elapsed:.2f}秒で処理 ({lines_per_second:.2f}行/秒)")

def process_multi_gpu(lines: list[str], output_file: str, batch_size: int = 16, num_gpus: int = None) -> None:
    """
    複数のGPUで処理を行う関数
    """
    # 利用可能なGPUの数を取得
    available_gpus = get_available_gpus()
    if not num_gpus:
        num_gpus = len(available_gpus)
    else:
        num_gpus = min(num_gpus, len(available_gpus))
    
    if num_gpus <= 1 or not torch.cuda.is_available():
        # GPUが1台以下の場合はシングルGPU処理
        return process_batch(lines, output_file, available_gpus[0], batch_size)
    
    # 各GPUに均等に行を分配
    chunks = [[] for _ in range(num_gpus)]
    for i, line in enumerate(lines):
        chunks[i % num_gpus].append(line)
    
    print(f"{num_gpus}台のGPUを使用して処理を開始: {[gpu for gpu in available_gpus[:num_gpus]]}")
    
    # 各GPUで処理した結果を保存する一時ファイル
    temp_files = [f"{output_file}.part{i}" for i in range(num_gpus)]
    
    # ProcessPoolExecutorを使用して並列処理
    with ProcessPoolExecutor(max_workers=num_gpus) as executor:
        futures = []
        for i in range(num_gpus):
            futures.append(
                executor.submit(
                    process_batch,
                    chunks[i],
                    temp_files[i],
                    available_gpus[i],
                    batch_size
                )
            )
        
        # すべての処理が完了するのを待つ
        for future in tqdm(as_completed(futures), total=num_gpus, desc="GPU処理中"):
            future.result()  # 例外があれば再度発生させる
    
    # 全ての一時ファイルを結合
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                with open(temp_file, 'r', encoding='utf-8') as infile:
                    outfile.write(infile.read())
                # 一時ファイルを削除
                os.remove(temp_file)
    
    print(f"全ての処理が完了しました")

@click.command()
@click.option(
    "--metadata",
    default="data/example/metadata.list",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
@click.option("--cleaned-path", default=None)
@click.option("--train-path", default=None)
@click.option("--val-path", default=None)
@click.option(
    "--config_path",
    default="configs/config.json",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
@click.option("--val-per-spk", default=4)
@click.option("--max-val-total", default=8)
@click.option("--clean/--no-clean", default=True)
@click.option(
    "--device",
    default="cuda:0",
    help="使用するGPUデバイス (マルチGPU使用時は無視されます)"
)
@click.option(
    "--batch-size",
    default=16,
    help="バッチサイズ"
)
@click.option(
    "--num-gpus",
    default=None,
    type=int,
    help="使用するGPUの数 (Noneの場合はすべて使用)"
)
def main(
    metadata: str,
    cleaned_path: str | None,
    train_path: str | None,
    val_path: str | None,
    config_path: str,
    val_per_spk: int,
    max_val_total: int,
    clean: bool,
    device: str,
    batch_size: int,
    num_gpus: int,
):
    if train_path is None:
        train_path = os.path.join(os.path.dirname(metadata), 'train.list')
    if val_path is None:
        val_path = os.path.join(os.path.dirname(metadata), 'val.list')
    if cleaned_path is None:
        cleaned_path = metadata + ".cleaned"

    if clean:
        # 入力ファイルを読み込む
        with open(metadata, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines()]
        
        # 複数GPUで処理を実行
        process_multi_gpu(lines, cleaned_path, batch_size, num_gpus)

    # スピーカー情報の処理
    print("スピーカー情報を処理中...")
    spk_utt_map = defaultdict(list)
    spk_id_map = {}
    current_sid = 0

    with open(cleaned_path, encoding="utf-8") as f:
        for line in tqdm(f.readlines(), desc="スピーカー情報読み込み中"):
            try:
                utt, spk, language, text, phones, tones, word2ph = line.strip().split("|")
                spk_utt_map[spk].append(line)
                if spk not in spk_id_map:
                    spk_id_map[spk] = current_sid
                    current_sid += 1
            except Exception as e:
                print(f"Error processing line: {line.strip()}\nError: {str(e)}")
                continue

    # データの分割
    train_list = []
    val_list = []
    
    print(f"スピーカー数: {len(spk_utt_map)}")
    for spk, utts in tqdm(spk_utt_map.items(), desc="データ分割中"):
        shuffle(utts)
        val_list += utts[:val_per_spk]
        train_list += utts[val_per_spk:]

    if len(val_list) > max_val_total:
        train_list += val_list[max_val_total:]
        val_list = val_list[:max_val_total]

    print(f"トレーニングデータ: {len(train_list)}行")
    print(f"検証データ: {len(val_list)}行")

    # データの保存
    with open(train_path, "w", encoding="utf-8") as f:
        for line in train_list:
            f.write(line)

    with open(val_path, "w", encoding="utf-8") as f:
        for line in val_list:
            f.write(line)

    # 設定ファイルの更新
    config = json.load(open(config_path, encoding="utf-8"))
    config["data"]["spk2id"] = spk_id_map
    config["data"]["training_files"] = train_path
    config["data"]["validation_files"] = val_path
    config["data"]["n_speakers"] = len(spk_id_map)
    config["num_languages"] = num_languages
    config["num_tones"] = num_tones
    config["symbols"] = symbols

    # 設定を保存
    config_save_path = os.path.join(os.path.dirname(metadata), 'config.json')
    with open(config_save_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"設定ファイルを保存しました: {config_save_path}")

def profile_main():
    """プロファイリング用のメイン関数"""
    profiler = cProfile.Profile()
    profiler.enable()
    
    # メイン処理の実行
    main()
    
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("cumtime")
    print("\n=== プロファイリング結果 ===")
    stats.print_stats(20)  # 上位20件を表示

if __name__ == "__main__":
    mp.set_start_method('spawn')
    profile_main()  # プロファイリング付きで実行