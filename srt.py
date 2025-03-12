import os
import time
import torch
import torchaudio
import whisper
from tqdm import tqdm
from silero_vad import (load_silero_vad, read_audio, get_speech_timestamps)

# === 設定 ===
SAMPLING_RATE = 16000  # Silero VAD は 16kHz で動作
AUDIO_DIR = "/home/abeto/parlertts_training_data/audio"  # 音声ファイルが入っているフォルダ
OUTPUT_DIR = "/home/abeto/parlertts_training_data/makesrt-kai"  # SRT ファイルを保存するフォルダ
SPLIT_AUDIO_BASE_DIR = "/home/abeto/parlertts_training_data/split_audio-kai/"  # 分割した音声を保存するフォルダ

# 出力フォルダを作成（存在しない場合のみ）
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SPLIT_AUDIO_BASE_DIR, exist_ok=True)

# === SRT 形式に変換する関数（ミリ秒単位対応） ===
def seconds_to_srt_time(seconds):
    """ 秒数を SRT 形式（HH:MM:SS,mmm）に変換 """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    sec = int(seconds % 60)
    milliseconds = int((seconds - int(seconds)) * 1000)  # ミリ秒を計算
    return f"{hours:02}:{minutes:02}:{sec:02},{milliseconds:03}"

# === Whisper のモデルをロード（large-v3, GPU使用） ===
device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("large-v3-turbo").to(device)

# 発話区間を統合する関数
def merge_speech_segments(speech_timestamps, merge_threshold=0.4, sampling_rate=16000):
    """
    連続する発話区間が `merge_threshold` 秒未満で離れている場合、それらを結合する。
    """
    if not speech_timestamps:
        return []

    merged_timestamps = [speech_timestamps[0]]

    for segment in speech_timestamps[1:]:
        prev_segment = merged_timestamps[-1]
        prev_end_sec = prev_segment["end"] / sampling_rate
        curr_start_sec = segment["start"] / sampling_rate

        # もし前の end との間隔が `merge_threshold` 秒未満なら結合
        if curr_start_sec - prev_end_sec <= merge_threshold:
            merged_timestamps[-1]["end"] = segment["end"]
        else:
            merged_timestamps.append(segment)

    return merged_timestamps

# === フォルダ内のすべての音声ファイルを処理 ===
audio_files = sorted([f for f in os.listdir(AUDIO_DIR) if f.lower().endswith((".wav", ".mp3", ".flac"))])
total_files = len(audio_files)

start_time = time.time()  # 全体の処理開始時間

for file_idx, audio_filename in enumerate(tqdm(audio_files, desc="Processing Audio Files", unit="file")):
    audio_path = os.path.join(AUDIO_DIR, audio_filename)
    base_name = os.path.splitext(audio_filename)[0]  # 拡張子を除いたファイル名
    split_audio_dir = os.path.join(SPLIT_AUDIO_BASE_DIR, base_name)
    output_srt_path = os.path.join(OUTPUT_DIR, f"{base_name}.srt")

    os.makedirs(split_audio_dir, exist_ok=True)  # ファイルごとの分割フォルダを作成

    print(f"\n=== {file_idx+1}/{total_files}: {audio_filename} を処理中 ===")

    # === Silero VAD で発話区間を取得し、結合処理を適用 ===
    vad_model = load_silero_vad(onnx=True)
    wav = read_audio(audio_path, sampling_rate=SAMPLING_RATE)
    speech_timestamps = get_speech_timestamps(wav, vad_model, sampling_rate=SAMPLING_RATE)
    merged_speech_timestamps = merge_speech_segments(speech_timestamps, merge_threshold=0.4, sampling_rate=SAMPLING_RATE)

    srt_content = []
    subtitle_index = 1

    # === 結合後の発話区間ごとに処理 ===
    start_file_time = time.time()  # ファイル処理の開始時間

    for i, segment in enumerate(tqdm(merged_speech_timestamps, desc="Processing Segments", unit="segment"), start=1):
        start_sample = segment["start"]
        end_sample = segment["end"]

        start_seconds = start_sample / SAMPLING_RATE
        end_seconds = end_sample / SAMPLING_RATE

        # 発話区間の音声を抽出し保存
        speech_chunk = wav[start_sample:end_sample]
        split_audio_path = os.path.join(split_audio_dir, f"chunk_{i:03d}.wav")
        torchaudio.save(split_audio_path, speech_chunk.unsqueeze(0), SAMPLING_RATE)

        # Whisper で文字起こし（日本語, GPU使用）
        result = model.transcribe(split_audio_path, language="ja")
        text = result["text"].strip()

        if text:
            start_time_str = seconds_to_srt_time(start_seconds)
            end_time_str = seconds_to_srt_time(end_seconds)
            srt_content.append(f"{subtitle_index}\n{start_time_str} --> {end_time_str}\n{text}\n")
            subtitle_index += 1

    # === SRT ファイルを保存 ===
    with open(output_srt_path, "w", encoding="utf-8") as f:
        f.writelines(srt_content)

    file_time_elapsed = time.time() - start_file_time
    avg_time_per_file = (time.time() - start_time) / (file_idx + 1)
    estimated_remaining = avg_time_per_file * (total_files - (file_idx + 1))

    print(f"  → {len(merged_speech_timestamps)} 個の発話区間を処理しました。")
    print(f"  → SRT ファイルを作成しました: {output_srt_path}")
    print(f"  → 所要時間: {file_time_elapsed:.2f} 秒")
    print(f"  → 推定残り時間: {estimated_remaining/60:.2f} 分")

# 全体の処理時間
total_time_elapsed = time.time() - start_time
print(f"\nすべての音声ファイルの処理が完了しました。")
print(f"総処理時間: {total_time_elapsed/60:.2f} 分")
