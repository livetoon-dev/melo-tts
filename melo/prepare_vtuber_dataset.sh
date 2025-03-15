#!/bin/bash

# データセット準備スクリプト
echo "VTuberデータセット準備開始"

# 仮想環境のパス
VENV_PATH="/home/nagashimadaichi/dev/melo-tts/.venv"

# データディレクトリ
DATA_DIR="/home/nagashimadaichi/dev/melo-tts/melo/data/vtuber"
METADATA="${DATA_DIR}/metadata.list"

# 実行ファイルの存在確認
if [ ! -f "$METADATA" ]; then
    echo "エラー: メタデータファイルが見つかりません: $METADATA"
    exit 1
fi

echo "===== ステップ1: テキストクリーニング ====="
$VENV_PATH/bin/python /home/nagashimadaichi/dev/melo-tts/melo/clean_text_data.py --metadata "$METADATA"
if [ $? -ne 0 ]; then
    echo "エラー: テキストクリーニングに失敗しました"
    exit 1
fi
CLEANED_TEXT="${METADATA}.txt_cleaned"

echo "===== ステップ2: 音声品質チェック ====="
$VENV_PATH/bin/python /home/nagashimadaichi/dev/melo-tts/melo/check_audio_quality.py --metadata "$CLEANED_TEXT"
if [ $? -ne 0 ]; then
    echo "エラー: 音声品質チェックに失敗しました"
    exit 1
fi
FILTERED_AUDIO="${CLEANED_TEXT}.audio_filtered"

echo "===== ステップ3: BERT特徴量抽出とデータセット分割 ====="
$VENV_PATH/bin/python /home/nagashimadaichi/dev/melo-tts/melo/preprocess_text.py --metadata "$FILTERED_AUDIO" --val-per-spk 8
if [ $? -ne 0 ]; then
    echo "エラー: BERT特徴量抽出に失敗しました"
    exit 1
fi

echo "データセット準備完了！"
echo "トレーニングするには次のコマンドを実行してください:"
echo "cd /home/nagashimadaichi/dev/melo-tts/melo && bash train_random_port.sh data/vtuber/config.json 1" 