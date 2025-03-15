import os
import librosa
import numpy as np
import click
from tqdm import tqdm

@click.command()
@click.option("--metadata", default="data/vtuber/metadata.list.txt_cleaned", help="クリーニング済みメタデータ")
@click.option("--output", default=None, help="フィルタリング後のメタデータ")
@click.option("--min-duration", default=0.5, help="最小音声長（秒）")
@click.option("--max-duration", default=10.0, help="最大音声長（秒）")
@click.option("--min-volume", default=0.005, help="最小音量閾値")
def main(metadata, output, min_duration, max_duration, min_volume):
    if output is None:
        output = metadata + ".audio_filtered"
    
    output_rejected = metadata + ".audio_rejected"
    output_stats = metadata + ".audio_stats"
    
    stats = {
        "total": 0,
        "kept": 0,
        "too_short": 0,
        "too_long": 0,
        "too_quiet": 0,
        "file_error": 0
    }
    
    print(f"音声品質チェック開始: {metadata}")
    
    with open(metadata, 'r', encoding='utf-8') as f_in, \
         open(output, 'w', encoding='utf-8') as f_out, \
         open(output_rejected, 'w', encoding='utf-8') as f_rej:
        
        for line in tqdm(f_in):
            stats["total"] += 1
            parts = line.strip().split('|')
            if len(parts) < 4:
                continue
                
            wavpath, spk, lang, text = parts[:4]
            
            # 音声をロード
            try:
                y, sr = librosa.load(wavpath, sr=None)
                duration = librosa.get_duration(y=y, sr=sr)
                rms = np.sqrt(np.mean(y**2))
                
                reject_reason = None
                if duration < min_duration:
                    stats["too_short"] += 1
                    reject_reason = "too_short"
                elif duration > max_duration:
                    stats["too_long"] += 1
                    reject_reason = "too_long"
                elif rms < min_volume:
                    stats["too_quiet"] += 1
                    reject_reason = "too_quiet"
                
                if reject_reason:
                    f_rej.write(f"{wavpath}|{spk}|{lang}|{text}|{reject_reason}|{duration:.2f}s|{rms:.6f}\n")
                else:
                    f_out.write(line)
                    stats["kept"] += 1
            except Exception as e:
                stats["file_error"] += 1
                f_rej.write(f"{wavpath}|{spk}|{lang}|{text}|error|{str(e)}\n")
    
    # 統計情報の保存
    with open(output_stats, 'w', encoding='utf-8') as f_stats:
        for key, value in stats.items():
            f_stats.write(f"{key}: {value}\n")
        
        if stats["total"] > 0:
            f_stats.write(f"保持率: {stats['kept']/stats['total']*100:.2f}%\n")
            f_stats.write(f"短すぎる音声率: {stats['too_short']/stats['total']*100:.2f}%\n")
            f_stats.write(f"長すぎる音声率: {stats['too_long']/stats['total']*100:.2f}%\n")
            f_stats.write(f"音量不足率: {stats['too_quiet']/stats['total']*100:.2f}%\n")
            f_stats.write(f"ファイルエラー率: {stats['file_error']/stats['total']*100:.2f}%\n")
    
    print(f"音声品質チェック完了: {stats['kept']} 件保持、{stats['total'] - stats['kept']} 件除外")
    print(f"詳細な統計は {output_stats} に保存されました")
    
    return output

if __name__ == "__main__":
    main() 