#!/bin/bash

# テスト用の前処理スクリプト
# 少量のデータで前処理をテストし、問題がなければ本番環境で実行します

# 環境変数の設定
BASE_DIR="/home/nagashimadaichi/dev/melo-tts/melo"
DATA_DIR="${BASE_DIR}/data/vtuber"

# 作業ディレクトリに移動
cd "${BASE_DIR}" || exit 1

# ステップ1: メタデータからテスト用のデータを作成（最初の10件）
echo "=== テスト用データの作成 ==="
if [ -f "${DATA_DIR}/metadata.list" ]; then
    head -n 10 "${DATA_DIR}/metadata.list" > "${DATA_DIR}/test_metadata.list"
    echo "テスト用メタデータを作成: ${DATA_DIR}/test_metadata.list (10件)"
else
    echo "エラー: メタデータファイルが見つかりません: ${DATA_DIR}/metadata.list"
    exit 1
fi

# ステップ2: テキストクリーニング
echo "=== テキストクリーニング ==="
uv run clean_text_data.py --metadata "${DATA_DIR}/test_metadata.list"
if [ $? -ne 0 ]; then
    echo "エラー: テキストクリーニングに失敗しました"
    exit 1
fi

# ステップ3: 音声品質チェック
echo "=== 音声品質チェック ==="
uv run check_audio_quality.py --metadata "${DATA_DIR}/test_metadata.list.txt_cleaned"
if [ $? -ne 0 ]; then
    echo "エラー: 音声品質チェックに失敗しました"
    exit 1
fi

# ステップ4: BERT特徴量抽出
echo "=== BERT特徴量抽出 ==="
uv run preprocess_text.py --metadata "${DATA_DIR}/test_metadata.list.txt_cleaned.audio_filtered" --num-processes 2
if [ $? -ne 0 ]; then
    echo "エラー: BERT特徴量抽出に失敗しました"
    exit 1
fi

echo "テスト前処理が完了しました！"
echo "テスト結果を確認してください。問題がなければ本番環境で実行できます。" 