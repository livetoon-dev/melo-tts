import os
import torch

# トレーニングリストのパス
train_list_path = "/home/nagashimadaichi/dev/melo-tts/melo/data/vtuber/train.list"
output_path = "/home/nagashimadaichi/dev/melo-tts/melo/data/vtuber/train.list.fixed"

# カウンター初期化
count_fixed_lang = 0
count_missing_bert = 0
count_missing_wav = 0
count_total = 0
count_valid = 0

# 言語コードを修正し、不足しているファイルを除外したリストを作成
with open(train_list_path, 'r') as f_in, open(output_path, 'w') as f_out:
    for line in f_in:
        count_total += 1
        parts = line.split('|')
        if len(parts) < 3:
            continue
        
        # ファイルパスを取得して両端のスペースを削除
        file_path = parts[0].strip()
        
        # ファイルパスとセパレータの間にスペースがあるかチェック
        if ' |' in file_path:
            # スペースとセパレータの位置を見つける
            space_pos = file_path.rfind(' |')
            # 実際のファイルパスを取得
            clean_path = file_path[:space_pos].strip()
            # 残りの部分を保持
            rest = file_path[space_pos+1:]
            file_path = clean_path
            # 修正された部分を更新
            parts[0] = file_path
        
        # WAVファイルが存在するか確認
        if not os.path.exists(file_path):
            print(f"WAVファイルが見つかりません: {file_path}")
            count_missing_wav += 1
            continue
            
        # BERTファイルが存在するか確認
        bert_path = file_path.replace(".wav", ".bert.pt")
        if not os.path.exists(bert_path):
            print(f"BERTファイルが見つかりません: {bert_path}")
            count_missing_bert += 1
            continue
            
        # 言語コードを修正
        if parts[2] == 'JA':
            parts[2] = 'JP'
            count_fixed_lang += 1
            
        # 有効なエントリとしてカウント
        count_valid += 1
        
        # 修正した行を書き込み
        fixed_line = '|'.join(parts)
        f_out.write(fixed_line)

# 新しいリストで元のリストを置き換え
os.rename(output_path, train_list_path)

print(f"=== 修正レポート ===")
print(f"元のエントリ数: {count_total}")
print(f"言語コード修正数: {count_fixed_lang}")
print(f"WAVファイル欠落数: {count_missing_wav}")
print(f"BERTファイル欠落数: {count_missing_bert}")
print(f"有効なエントリ数: {count_valid}")
print(f"===========================")
print(f"処理完了！新しいトレーニングリストを作成しました。") 