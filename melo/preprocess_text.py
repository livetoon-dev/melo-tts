import json
from collections import defaultdict
from random import shuffle
from typing import Optional, List, Tuple
import multiprocessing
from functools import partial
import os
import torch
from tqdm import tqdm
import click
from text.cleaner import clean_text_bert
import os
import torch
from text.symbols import symbols, num_languages, num_tones

def process_line(line: str, clean: bool, use_sudachi: bool, gpu_id: int) -> Optional[Tuple[str, str, str, str, str, str, str, str]]:
    """1行のデータを処理する関数"""
    try:
        utt, spk, language, text = line.strip().split("|")
        print(f"Processing: {utt}, language: {language}, text: {text}")
        try:
            # GPUデバイスを指定
            device = f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'
            
            # BERT特徴抽出の前にモデル初期化を明示的に行う
            if language.upper() in ["JA", "JP"]:
                from style_bert_vits2.constants import Languages
                from style_bert_vits2.nlp import bert_models
                
                # LINE DistilBERTモデルを事前に読み込む
                try:
                    bert_models.load_model(Languages.JP, "line-corporation/line-distilbert-base-japanese", trust_remote_code=True)
                    bert_models.load_tokenizer(Languages.JP, "line-corporation/line-distilbert-base-japanese", trust_remote_code=True)
                except Exception as e:
                    print(f"BERT model initialization failed: {str(e)}")
            
            # テキスト処理とBERT特徴抽出
            print(f"Calling clean_text_bert with use_sudachi={use_sudachi}")
            norm_text, phones, tones, word2ph, bert = clean_text_bert(text, language, device=device, use_sudachi=use_sudachi)
            
            # 音素チェック
            new_symbols = []
            for ph in phones:
                if ph not in symbols and ph not in new_symbols:
                    new_symbols.append(ph)
                    print('update!, now symbols:')
                    print(new_symbols)
                    with open(f'{language}_symbol.txt', 'w') as f:
                        f.write(f'{new_symbols}')

            assert len(phones) == len(tones)
            assert len(phones) == sum(word2ph)
            
            # BERT特徴量を保存
            bert_path = utt.replace(".wav", ".bert.pt")
            os.makedirs(os.path.dirname(bert_path), exist_ok=True)
            torch.save(bert, bert_path)
            print(f"Saved BERT features to {bert_path}")
            
            return (
                utt,
                spk,
                language,
                norm_text,
                " ".join(phones),
                " ".join([str(i) for i in tones]),
                " ".join([str(i) for i in word2ph]),
                line
            )
        except Exception as e:
            print(f"Error processing text: {text}")
            print(f"Error details: {str(e)}")
            return None
    except Exception as e:
        print(f"Error parsing line: {line.strip()}")
        print(f"Error details: {str(e)}")
        return None

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
@click.option("--use-sudachi/--no-sudachi", default=True, help="日本語のテキスト処理にSudachiを使用する")
@click.option("--num-processes", default=8, help="並列処理に使用するプロセス数")
def main(
    metadata: str,
    cleaned_path: Optional[str],
    train_path: str,
    val_path: str,
    config_path: str,
    val_per_spk: int,
    max_val_total: int,
    clean: bool,
    use_sudachi: bool,
    num_processes: int,
):
    if train_path is None:
        train_path = os.path.join(os.path.dirname(metadata), 'train.list')
    if val_path is None:
        val_path = os.path.join(os.path.dirname(metadata), 'val.list')
    out_config_path = os.path.join(os.path.dirname(metadata), 'config.json')

    if cleaned_path is None:
        cleaned_path = metadata + ".cleaned"

    if clean:
        # 利用可能なGPUの数を確認
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        print(f"利用可能なGPU数: {num_gpus}")
        
        # プロセス数を調整（GPUの数を超えないようにする）
        if num_gpus > 0:
            num_processes = min(num_processes, num_gpus)
        
        print(f"並列処理に使用するプロセス数: {num_processes}")
        
        # 入力ファイルの行を読み込む
        lines = open(metadata, encoding="utf-8").readlines()
        
        if num_processes > 1 and num_gpus > 0:
            # 各GPUに割り当てるデータを分割
            chunks = [[] for _ in range(num_processes)]
            for i, line in enumerate(lines):
                chunks[i % num_processes].append((line, i % num_gpus))
            
            # 各GPUで並列処理
            results = []
            for gpu_id in range(min(num_processes, num_gpus)):
                chunk_results = []
                print(f"GPU {gpu_id} で {len(chunks[gpu_id])} 件のデータを処理します")
                
                for line, _ in tqdm(chunks[gpu_id], desc=f"GPU {gpu_id} 処理中"):
                    # 直接process_line関数を呼び出す形に修正
                    result = process_line(line, clean, use_sudachi, gpu_id)  # use_sudachi引数を正しく渡す
                    if result:
                        chunk_results.append(result)
                
                results.extend(chunk_results)
        else:
            # シングルプロセス処理
            results = []
            for line in tqdm(lines):
                result = process_line(line, clean, use_sudachi, 0)  # use_sudachi引数を正しく渡す
                if result:
                    results.append(result)
        
        # 結果を書き込む
        with open(cleaned_path, "w", encoding="utf-8") as out_file:
            for result in results:
                if result:
                    utt, spk, language, norm_text, phones, tones, word2ph, _ = result
                    out_file.write(
                        "{}|{}|{}|{}|{}|{}|{}\n".format(
                            utt, spk, language, norm_text, phones, tones, word2ph
                        )
                    )
        
        metadata = cleaned_path

    spk_utt_map = defaultdict(list)
    spk_id_map = {}
    current_sid = 0

    with open(metadata, encoding="utf-8") as f:
        for line in f.readlines():
            utt, spk, language, text, phones, tones, word2ph = line.strip().split("|")
            spk_utt_map[spk].append(line)

            if spk not in spk_id_map.keys():
                spk_id_map[spk] = current_sid
                current_sid += 1

    train_list = []
    val_list = []

    for spk, utts in spk_utt_map.items():
        shuffle(utts)
        val_list += utts[:val_per_spk]
        train_list += utts[val_per_spk:]

    if len(val_list) > max_val_total:
        train_list += val_list[max_val_total:]
        val_list = val_list[:max_val_total]

    with open(train_path, "w", encoding="utf-8") as f:
        for line in train_list:
            f.write(line)

    with open(val_path, "w", encoding="utf-8") as f:
        for line in val_list:
            f.write(line)

    config = json.load(open(config_path, encoding="utf-8"))
    config["data"]["spk2id"] = spk_id_map

    config["data"]["training_files"] = train_path
    config["data"]["validation_files"] = val_path
    config["data"]["n_speakers"] = len(spk_id_map)
    config["num_languages"] = num_languages
    config["num_tones"] = num_tones
    config["symbols"] = symbols

    with open(out_config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
