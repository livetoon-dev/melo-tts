import os
import json
import torch
import click
from tqdm import tqdm
import torch.multiprocessing as mp
from functools import partial
from collections import defaultdict
from random import shuffle
from text.cleaner import clean_text_bert
from text.symbols import symbols, num_languages, num_tones

# GPU 개수 자동 탐지 (최대 8개)
NUM_GPUS = min(8, torch.cuda.device_count())  
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
        with open(metadata, encoding="utf-8") as f:
            lines = f.readlines()

        print(f"🚀 {len(lines)} 개의 라인을 처리 중...")

        os.makedirs("tmp_cleaned", exist_ok=True)  # ✅ 임시 저장 폴더 생성

        # GPU를 활용한 병렬 처리
        with mp.Pool(NUM_GPUS) as pool:
            process_func = partial(process_line, device_list=[f'cuda:{i}' for i in range(NUM_GPUS)])
            list(tqdm(pool.imap(process_func, enumerate(lines)), total=len(lines)))

        print(f"📌 데이터 병합 중...")

        # ✅ 각 프로세스가 개별 파일에 저장한 데이터를 합치기
        with open(cleaned_path, "w", encoding="utf-8") as f_out:
            for i in range(NUM_GPUS):
                tmp_file = f"tmp_cleaned/cleaned_{i}.txt"
                if os.path.exists(tmp_file):
                    with open(tmp_file, "r", encoding="utf-8") as f_in:
                        f_out.write(f_in.read())
                    os.remove(tmp_file)  # ✅ 병합 후 삭제

        print(f"✅ 파일 저장 완료: {cleaned_path}")
        metadata = cleaned_path

    # 이후 train/val 분할 및 설정 저장
    process_metadata(metadata, train_path, val_path, config_path, out_config_path, val_per_spk, max_val_total)


def process_line(data, device_list):
    """
    개별 문장을 처리하여 음소 변환 및 BERT feature 추출.
    device_list에서 하나를 선택하여 GPU 병렬 처리.
    """
    import random
    index, line = data
    device = random.choice(device_list)  # 랜덤한 GPU 선택
    process_id = index % NUM_GPUS  # ✅ 프로세스별 파일 지정

    try:
        utt, spk, language, text = line.strip().split("|")
        norm_text, phones, tones, word2ph, bert = clean_text_bert(text, language, device=device)

        assert len(phones) == len(tones)
        assert len(phones) == sum(word2ph)

        bert_path = utt.replace(".wav", ".bert.pt")
        os.makedirs(os.path.dirname(bert_path), exist_ok=True)
        torch.save(bert.cpu(), bert_path)

        # ✅ 각 프로세스별로 파일 분리 저장
        with open(f"tmp_cleaned/cleaned_{process_id}.txt", "a", encoding="utf-8") as f:
            f.write(f"{utt}|{spk}|{language}|{norm_text}|{' '.join(phones)}|{' '.join(map(str, tones))}|{' '.join(map(str, word2ph))}\n")

    except Exception as error:
        print(f"❌ 오류 발생: {line} -> {error}")


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