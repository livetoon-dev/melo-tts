import re
import os
import click
from tqdm import tqdm

@click.command()
@click.option("--metadata", default="data/vtuber/metadata.list", help="元のメタデータファイル")
@click.option("--output", default=None, help="出力先（デフォルトは元ファイル名+.txt_cleaned）")
@click.option("--min-chars", default=3, help="最小文字数（これ未満は別ファイルに保存）")
def main(metadata, output, min_chars):
    if output is None:
        output = metadata + ".txt_cleaned"
    
    output_short = metadata + ".short_texts"  # 短すぎるテキスト用
    output_stats = metadata + ".cleaning_stats"  # 統計情報用
    
    # カウンタの初期化
    stats = {
        "total": 0,
        "kept": 0,
        "short": 0,
        "punct_fixed": 0,
        "ellipsis_fixed": 0,
        "exclamation_fixed": 0
    }
    
    print(f"テキストクリーニング開始: {metadata}")
    
    with open(metadata, 'r', encoding='utf-8') as f_in, \
         open(output, 'w', encoding='utf-8') as f_out, \
         open(output_short, 'w', encoding='utf-8') as f_short:
        
        for line in tqdm(f_in):
            stats["total"] += 1
            parts = line.strip().split('|')
            if len(parts) < 4:
                continue
                
            wavpath, spk, lang, text = parts[:4]
            
            # テキストのクリーニング
            cleaned_text, cleaning_info = clean_text(text)
            
            # 句読点修正があったかチェック
            if "punct_fixed" in cleaning_info:
                stats["punct_fixed"] += 1
            if "ellipsis_fixed" in cleaning_info:
                stats["ellipsis_fixed"] += 1
            if "exclamation_fixed" in cleaning_info:
                stats["exclamation_fixed"] += 1
            
            # 短いテキストのチェック
            if len(cleaned_text) < min_chars:
                f_short.write(f"{wavpath}|{spk}|{lang}|{cleaned_text}\n")
                stats["short"] += 1
            else:
                f_out.write(f"{wavpath}|{spk}|{lang}|{cleaned_text}\n")
                stats["kept"] += 1
    
    # 統計情報の保存
    with open(output_stats, 'w', encoding='utf-8') as f_stats:
        for key, value in stats.items():
            f_stats.write(f"{key}: {value}\n")
        
        if stats["total"] > 0:
            f_stats.write(f"短いテキスト率: {stats['short']/stats['total']*100:.2f}%\n")
            f_stats.write(f"句読点修正率: {stats['punct_fixed']/stats['total']*100:.2f}%\n")
            f_stats.write(f"三点リーダー修正率: {stats['ellipsis_fixed']/stats['total']*100:.2f}%\n")
            f_stats.write(f"感嘆符修正率: {stats['exclamation_fixed']/stats['total']*100:.2f}%\n")
    
    print(f"処理完了: {stats['kept']} 件保持, {stats['short']} 件短いテキスト")
    print(f"詳細な統計は {output_stats} に保存されました")
    
    return output

def clean_text(text):
    """テキストクリーニング関数"""
    cleaning_info = []
    original_text = text
    
    # 半角を全角に変換
    text = text.replace(',', '、').replace('.', '。')
    text = text.replace('!', '！').replace('?', '？')
    
    if text != original_text:
        cleaning_info.append("punct_fixed")
    
    # 三点リーダーを統一
    original_text = text
    text = re.sub(r'\.\.\.', '…', text)
    if text != original_text:
        cleaning_info.append("ellipsis_fixed")
    
    # 連続する感嘆符・疑問符を減らす (最大2つまで)
    original_text = text
    text = re.sub(r'！{3,}', '！！', text)
    text = re.sub(r'？{3,}', '？？', text)
    if text != original_text:
        cleaning_info.append("exclamation_fixed")
    
    # 余分な空白を削除
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text, cleaning_info

if __name__ == "__main__":
    main() 