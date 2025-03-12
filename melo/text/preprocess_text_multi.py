import os
import json
import torch
import click
from tqdm import tqdm
from multiprocessing import Pool, Manager
from functools import partial
from collections import defaultdict
from random import shuffle
from text.cleaner import clean_text_bert
from text.symbols import symbols, num_languages, num_tones

# GPU 개수 자동 탐지 (최대 4개)
NUM_GPUS = min(4, torch.cuda.device_count())  
assert NUM_GPUS > 0, "GPU를 찾을 수 없습니다!"

@click.command()
@click.option("--metadata", default="data/example/metadata.list", type=click.Path(exists=True))
@click.option("--cleaned-path", default=None)
@click.option("--train-path", default=None)
@click.option("--val-path", default=None)
@click.option("--config-path", default="configs/config.json", type=click.Path(exists=True))
@click.option("--val-per-spk", default=4)
@click.option("--max-val-total", default=8)
@click.option("--clean/--no-clean", default=True)
def main(metadata, cleaned_path, train_path, val_path, config_path, val_per_spk, max_val_total, clean):
    if train_path is None:
        train_path = os.path.join(os.path.dirname(metadata), 'train.list')
    if val_path is None:
        val_path = os.path.join(os.path.dirname(metadata), 'val.list')
    out_config_path = os.path.join(os.path.dirname(metadata), 'config.json')

    if cleaned_path is None:
        cleaned_path = metadata + ".cleaned"

    if clean:
        lines = open(metadata, encoding="utf-8").readlines()

        # 멀티 프로세싱을 위해 Manager 사용
        manager = Manager()
        result_queue = manager.Queue()

        # GPU 4개를 활용한 병렬 처리
        with Pool(NUM_GPUS) as pool:
            process_func = partial(process_line, device_list=[f'cuda:{i}' for i in range(NUM_GPUS)], result_queue=result_queue)
            results = list(tqdm(pool.imap(process_func, lines), total=len(lines)))

        # 처리된 데이터를 저장
        with open(cleaned_path, "w", encoding="utf-8") as f:
            while not result_queue.empty():
                line = result_queue.get()
                if line:
                    f.write(line + "\n")

        metadata = cleaned_path

    # 이후 train/val 분할 및 설정 저장
    process_metadata(metadata, train_path, val_path, config_path, out_config_path, val_per_spk, max_val_total)

def process_line(line, device_list, result_queue):
    """
    개별 문장을 처리하여 음소 변환 및 BERT feature 추출.
    device_list에서 하나를 선택하여 GPU 병렬 처리.
    """
    import random
    device = random.choice(device_list)  # 랜덤한 GPU 선택

    try:
        utt, spk, language, text = line.strip().split("|")
        norm_text, phones, tones, word2ph, bert = clean_text_bert(text, language, device=device)

        assert len(phones) == len(tones)
        assert len(phones) == sum(word2ph)

        bert_path = utt.replace(".wav", ".bert.pt")
        os.makedirs(os.path.dirname(bert_path), exist_ok=True)
        torch.save(bert.cpu(), bert_path)

        result_queue.put(f"{utt}|{spk}|{language}|{norm_text}|{' '.join(phones)}|{' '.join(map(str, tones))}|{' '.join(map(str, word2ph))}")
    except Exception as error:
        print(f"Error processing line: {line} -> {error}")

def process_metadata(metadata, train_path, val_path, config_path, out_config_path, val_per_spk, max_val_total):
    """
    전처리된 metadata에서 train/val 분할 및 설정 저장.
    """
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