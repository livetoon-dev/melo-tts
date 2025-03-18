import json
from collections import defaultdict
from random import shuffle
from typing import Optional, List, Tuple
import os
import torch
from tqdm import tqdm
import click
from text.cleaner import clean_text_bert
from text.symbols import symbols, num_languages, num_tones
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import threading

# 글로벌 변수로 new_symbols 리스트와 락 설정
new_symbols_lock = threading.Lock()
new_symbols = []

def process_line(line: str, language_file_prefix: str, device: str = 'cuda:0'):
    try:
        utt, spk, language, text = line.strip().split("|")
        norm_text, phones, tones, word2ph, bert = clean_text_bert(text, language, device=device)
        
        # 새로운 심볼 검출 및 기록
        global new_symbols
        new_symbols_to_add = []
        for ph in phones:
            if ph not in symbols and ph not in new_symbols:
                new_symbols_to_add.append(ph)
                
        if new_symbols_to_add:
            with new_symbols_lock:
                for ph in new_symbols_to_add:
                    if ph not in new_symbols:
                        new_symbols.append(ph)
                        print(f'새 심볼 추가: {ph}, 현재 심볼 개수: {len(new_symbols)}')
                with open(f'{language}_symbol.txt', 'w') as f:
                    f.write(f'{new_symbols}')

        assert len(phones) == len(tones)
        assert len(phones) == sum(word2ph)
        
        bert_path = utt.replace(".wav", ".bert.pt")
        os.makedirs(os.path.dirname(bert_path), exist_ok=True)
        torch.save(bert.cpu(), bert_path)
        
        return (utt, spk, language, norm_text, phones, tones, word2ph, None)
    except Exception as error:
        print(f"에러 발생! {line}, {error}")
        return None

def chunks(lst, n):
    """리스트를 n개의 청크로 분할"""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

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
@click.option("--num-workers", default=48, help="멀티프로세싱 작업자 수")
def main(
    metadata: str,
    cleaned_path: Optional[str],
    train_path: str,
    val_path: str,
    config_path: str,
    val_per_spk: int,
    max_val_total: int,
    clean: bool,
    num_workers: int,
):
    if train_path is None:
        train_path = os.path.join(os.path.dirname(metadata), 'train.list')
    if val_path is None:
        val_path = os.path.join(os.path.dirname(metadata), 'val.list')
    out_config_path = os.path.join(os.path.dirname(metadata), 'config.json')

    if cleaned_path is None:
        cleaned_path = metadata + ".cleaned"

    if clean:
        # CUDA 장치 분배
        available_cuda_devices = [f'cuda:{i}' for i in range(torch.cuda.device_count())]
        if not available_cuda_devices:
            available_cuda_devices = ['cpu']
            print("CUDA 장치가 없습니다. CPU를 사용합니다.")
        
        # 실제 작업자 수 설정
        num_workers = min(num_workers, multiprocessing.cpu_count())
        print(f"작업자 수: {num_workers}, 사용 가능한 CUDA 장치: {available_cuda_devices}")
        
        # 모든 라인 읽기
        lines = open(metadata, encoding="utf-8").readlines()
        
        # 결과 저장용 리스트
        results = []
        
        # 프로세스 풀 실행
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # 각 작업자에 디바이스 할당 (라운드 로빈 방식)
            futures = []
            for i, line in enumerate(lines):
                device = available_cuda_devices[i % len(available_cuda_devices)]
                futures.append(
                    executor.submit(
                        process_line, 
                        line, 
                        os.path.dirname(metadata),
                        device
                    )
                )
            
            # 결과 수집
            for future in tqdm(futures, desc="텍스트 전처리 중"):
                result = future.result()
                if result is not None:
                    results.append(result)
        
        # 결과 파일 작성
        with open(cleaned_path, "w", encoding="utf-8") as out_file:
            for (utt, spk, language, norm_text, phones, tones, word2ph, _) in results:
                out_file.write(
                    "{}|{}|{}|{}|{}|{}|{}\n".format(
                        utt,
                        spk,
                        language,
                        norm_text,
                        " ".join(phones),
                        " ".join([str(i) for i in tones]),
                        " ".join([str(i) for i in word2ph]),
                    )
                )
                
        print(f"전처리 완료! 총 {len(results)}개의 항목이 처리되었습니다.")
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