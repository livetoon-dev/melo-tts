import os
import sys
import json
import torch
import argparse
from pathlib import Path

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# BERT特徴量の生成に必要なモジュールをインポート
from style_bert_vits2.nlp.japanese import JapanBertFeatures
from melo.text.japanese import g2p_ja

# 設定ファイルを読み込む
def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# トレーニングリストからファイルリストを抽出
def get_files_from_train_list(train_list_path):
    files = []
    with open(train_list_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) >= 4:  # パス|話者|言語|テキスト
                wav_path = parts[0]
                text = parts[3]
                files.append((wav_path, text))
    return files

# メイン処理
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/config.json")
    parser.add_argument("--train_list", type=str, default="data/vtuber/train.list")
    args = parser.parse_args()

    # 設定を読み込む
    config = load_config(args.config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # BERT特徴量抽出器を初期化
    bert_model = JapanBertFeatures(config["bert"]["japanese"]).to(device)
    
    # ファイルリストを取得
    files = get_files_from_train_list(args.train_list)
    print(f"処理するファイル数: {len(files)}")
    
    # 各ファイルに対してBERT特徴量を生成
    success_count = 0
    for i, (wav_path, text) in enumerate(files):
        try:
            # BERT特徴量ファイルのパスを生成
            bert_path = wav_path.replace(".wav", ".bert.pt")
            
            # すでに存在する場合はスキップ
            if os.path.exists(bert_path):
                print(f"[{i+1}/{len(files)}] スキップ（すでに存在）: {bert_path}")
                success_count += 1
                continue
            
            # テキストを処理
            phones, tones = g2p_ja(text)
            bert = bert_model(text, device)
            
            # BERT特徴量を保存
            os.makedirs(os.path.dirname(bert_path), exist_ok=True)
            torch.save(bert, bert_path)
            
            success_count += 1
            print(f"[{i+1}/{len(files)}] 成功: {bert_path}")
            
            # 100ファイルごとに進捗を表示
            if (i + 1) % 100 == 0:
                print(f"進捗: {i+1}/{len(files)} ({(i+1)/len(files)*100:.2f}%)")
                
        except Exception as e:
            print(f"[{i+1}/{len(files)}] エラー: {wav_path} - {e}")
    
    print(f"処理完了: {success_count}/{len(files)} ファイルのBERT特徴量を生成しました。")

if __name__ == "__main__":
    main() 