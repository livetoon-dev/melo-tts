#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train.listから存在しないWAVファイルの参照を削除するスクリプト
"""

import os
import sys
from tqdm import tqdm

def clean_list_file(list_file, output_file=None):
    """
    リストファイルから存在しないファイルの参照を削除します。
    
    Args:
        list_file (str): 処理するリストファイルのパス
        output_file (str, optional): 出力ファイルのパス。Noneの場合は元のファイルに上書きします。
    
    Returns:
        int: 削除された行数
    """
    if output_file is None:
        output_file = list_file + ".cleaned"
    
    lines = []
    invalid_lines = []
    
    print(f"リストファイル {list_file} を読み込み中...")
    with open(list_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"合計 {len(lines)} 行を読み込みました。")
    print("無効なファイル参照をチェック中...")
    
    valid_lines = []
    
    for i, line in enumerate(tqdm(lines)):
        parts = line.strip().split('|')
        if not parts:
            continue
        
        wav_path = parts[0]
        if os.path.exists(wav_path):
            valid_lines.append(line)
        else:
            invalid_lines.append((i+1, wav_path))
    
    print(f"無効なファイル参照が {len(invalid_lines)} 個見つかりました。")
    
    if invalid_lines:
        print("最初の10個の無効なファイル参照:")
        for i, (line_num, path) in enumerate(invalid_lines[:10]):
            print(f"{i+1}. 行 {line_num}: {path}")
    
    print(f"有効な行 {len(valid_lines)} 行を {output_file} に書き込み中...")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(valid_lines)
    
    print(f"完了しました。{len(invalid_lines)} 行が削除されました。")
    
    return len(invalid_lines)

def main():
    if len(sys.argv) < 2:
        print(f"使用方法: {sys.argv[0]} <list_file> [output_file]")
        sys.exit(1)
    
    list_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(list_file):
        print(f"エラー: ファイル {list_file} が見つかりません。")
        sys.exit(1)
    
    clean_list_file(list_file, output_file)

if __name__ == "__main__":
    main() 