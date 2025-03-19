import json
import os
import torch
import multiprocessing as mp
from collections import defaultdict
from random import shuffle
from tqdm import tqdm
import click
from text.cleaner import clean_text_bert
from text.symbols import symbols, num_languages, num_tones

def process_file(file_chunk, cleaned_path, gpu_id, temp_dir):
    out_file = open(os.path.join(temp_dir, f"{os.path.basename(cleaned_path)}.{gpu_id}"), "w", encoding="utf-8")
    new_symbols = []
    device = f"cuda:{gpu_id}"
    
    for line in tqdm(open(file_chunk, encoding="utf-8").readlines()):
        try:
            utt, spk, language, text = line.strip().split("|")
            norm_text, phones, tones, word2ph, bert = clean_text_bert(text, language, device=device)
            
            for ph in phones:
                if ph not in symbols and ph not in new_symbols:
                    new_symbols.append(ph)
                    with open(f'{language}_symbol.txt', 'w') as f:
                        f.write(f'{new_symbols}')
            
            assert len(phones) == len(tones) == sum(word2ph)
            out_file.write(
                f"{utt}|{spk}|{language}|{norm_text}|{' '.join(phones)}|{' '.join(map(str, tones))}|{' '.join(map(str, word2ph))}\n"
            )
            
            bert_path = utt.replace(".wav", ".bert.pt")
            os.makedirs(os.path.dirname(bert_path), exist_ok=True)
            torch.save(bert.cpu(), bert_path)
        except Exception as error:
            print("err!", line, error)
    
    out_file.close()

def split_file(metadata, num_chunks=8):
    """텍스트 파일을 8개로 분할"""
    with open(metadata, encoding="utf-8") as f:
        lines = f.readlines()
    
    shuffle(lines)
    chunk_size = len(lines) // num_chunks
    chunks = [lines[i * chunk_size:(i + 1) * chunk_size] for i in range(num_chunks)]
    if len(lines) % num_chunks:
        chunks[-1].extend(lines[num_chunks * chunk_size:])
    
    chunk_files = []
    for i, chunk in enumerate(chunks):
        chunk_path = f"{metadata}.chunk{i}"
        with open(chunk_path, "w", encoding="utf-8") as f:
            f.writelines(chunk)
        chunk_files.append(chunk_path)
    
    return chunk_files

@click.command()
@click.option("--metadata", default="data/example/metadata.list", type=click.Path(exists=True))
@click.option("--cleaned-path", default=None)
@click.option("--temp-dir", default="temp_files", help="임시 파일 저장 경로")
@click.option("--num-chunks", default=8, help="병렬 실행할 프로세스 개수")
def main(metadata, cleaned_path, temp_dir, num_chunks):
    if cleaned_path is None:
        cleaned_path = metadata + ".cleaned"
    
    os.makedirs(temp_dir, exist_ok=True)
    chunk_files = split_file(metadata, num_chunks)
    processes = []
    
    for i, chunk_file in enumerate(chunk_files):
        p = mp.Process(target=process_file, args=(chunk_file, cleaned_path, i, temp_dir))
        processes.append(p)
        p.start()
    
    for p in processes:
        p.join()
    
    # 결과 병합
    with open(cleaned_path, "w", encoding="utf-8") as out_file:
        for i in range(num_chunks):
            with open(os.path.join(temp_dir, f"{os.path.basename(cleaned_path)}.{i}"), encoding="utf-8") as f:
                out_file.writelines(f.readlines())

if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
