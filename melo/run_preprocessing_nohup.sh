#!/bin/bash
set -e

# 作業ディレクトリの設定
cd "$(dirname "$0")"

# VTuberデータセットの準備
echo "VTuberデータセット準備開始"

# 以前の処理結果をクリーンアップ
echo "以前の処理結果をクリーンアップしています..."
find data/wav -name "*.bert.pt" -delete
rm -f data/vtuber/metadata.list.txt_cleaned data/vtuber/metadata.list.txt_cleaned.audio_filtered data/vtuber/metadata.list.cleaning_stats data/vtuber/metadata.list.txt_cleaned.audio_stats

# バックグラウンドで処理を開始
echo "前処理を開始します..."
nohup ./prepare_vtuber_dataset.sh > preprocessing.log 2>&1 &

# プロセスIDを表示
PID=$!
echo "前処理がバックグラウンドで開始されました（プロセスID: $PID）"
echo "処理の進行状況を確認するには、以下のコマンドを実行してください:"
echo "tail -f preprocessing.log"

# ステップ1: テキストクリーニング
echo "===== ステップ1: テキストクリーニング ====="
echo "テキストクリーニング開始: $(pwd)/data/vtuber/metadata.list"
python -m vtuber_tools.filter_short_text --input_file data/vtuber/metadata.list --output_file data/vtuber/metadata.list.txt_cleaned --min_length 3

# ステップ2: 音声品質チェック
echo "===== ステップ2: 音声品質チェック ====="
echo "音声品質チェック開始: $(pwd)/data/vtuber/metadata.list.txt_cleaned"
python -m vtuber_tools.filter_audio_quality --input_file data/vtuber/metadata.list.txt_cleaned --output_file data/vtuber/metadata.list.txt_cleaned.audio_filtered

# ステップ3: 前処理
echo "===== ステップ3: テキスト前処理 ====="
echo "テキスト前処理開始: $(pwd)/data/vtuber/metadata.list.txt_cleaned.audio_filtered"
# Sudachiを使用するために--no-sudachiオプションを削除
uv run preprocess_text.py --metadata data/vtuber/metadata.list.txt_cleaned.audio_filtered --num-processes 8 > preprocessing.log 2>&1

echo "前処理が完了しました。" 