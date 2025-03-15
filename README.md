<div align="center">
  <div>&nbsp;</div>
  <img src="logo.png" width="300"/> <br>
  <a href="https://trendshift.io/repositories/8133" target="_blank"><img src="https://trendshift.io/api/badge/repositories/8133" alt="myshell-ai%2FMeloTTS | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</div>

# MeloTTS 日本語対応強化版

## 主な改良点

- **LINE DistilBERT対応**  
  日本語処理に特化したLINE DistilBERT（768次元）に対応しました。これにより、日本語音声合成の品質が向上します。

- **日本語言語コード対応の強化**  
  言語コードとして「JA」と「JP」の両方に対応しました。これにより、どちらの表記を使用しても正しく日本語として認識されます。

- **エラーハンドリングの改善**  
  未対応の言語コードに対して適切なエラーメッセージを表示するようになりました。

## 使い方

1. 環境構築
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   uv sync
   ```

2. 辞書のダウンロード
   ```bash
   cd melo
   uv run python -m unidic download
   uv pip install pyopenjtalk-plus
   ```

3. トレーニング実行
   ```bash
   cd melo
   bash train.sh data/your_config.json 1
   ```

## 元のMeloTTSとの差分

- LINE DistilBERT（768次元）のサポート追加
- 言語コード「JA」のサポート追加（元は「JP」のみ）
- エラーハンドリングの強化
- マルチプロセス設定の最適化
- ポート競合問題の解決（ランダムポート対応）

## レポジトリについて

- **ソースについて**  
  このプロジェクトはMeloTTSからコピーしています。

- **日本語対応の改善**  
  日本語向けに改善を行っています。詳細は[こちら](https://zenn.dev/kun432/scraps/34d9ff1874bd3b)をご参照ください。

- **プロジェクト管理 (uv利用)**  
  プロジェクトは[uv](https://astral.sh/uv)で管理しています。環境構築は以下のコマンドを実行してください:
  
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  uv sync
  ```

- 辞書のダウンロード (トレーニング前に必要)
トレーニング前に、以下のコマンドで辞書をダウンロードしてください:

  ```bash
  cd melo
  uv run python -m unidic download
  uv pip install pyopenjtalk-plus
  ```

- データ配置について
データは melo/data 配下に配置してください。


## Introduction
MeloTTS is a **high-quality multi-lingual** text-to-speech library by [MIT](https://www.mit.edu/) and [MyShell.ai](https://myshell.ai). Supported languages include:

| Language | Example |
| --- | --- |
| English (American)    | [Link](https://myshell-public-repo-host.s3.amazonaws.com/myshellttsbase/examples/en/EN-US/speed_1.0/sent_000.wav) |
| English (British)     | [Link](https://myshell-public-repo-host.s3.amazonaws.com/myshellttsbase/examples/en/EN-BR/speed_1.0/sent_000.wav) |
| English (Indian)      | [Link](https://myshell-public-repo-host.s3.amazonaws.com/myshellttsbase/examples/en/EN_INDIA/speed_1.0/sent_000.wav) |
| English (Australian)  | [Link](https://myshell-public-repo-host.s3.amazonaws.com/myshellttsbase/examples/en/EN-AU/speed_1.0/sent_000.wav) |
| English (Default)     | [Link](https://myshell-public-repo-host.s3.amazonaws.com/myshellttsbase/examples/en/EN-Default/speed_1.0/sent_000.wav) |
| Spanish               | [Link](https://myshell-public-repo-host.s3.amazonaws.com/myshellttsbase/examples/es/ES/speed_1.0/sent_000.wav) |
| French                | [Link](https://myshell-public-repo-host.s3.amazonaws.com/myshellttsbase/examples/fr/FR/speed_1.0/sent_000.wav) |
| Chinese (mix EN)      | [Link](https://myshell-public-repo-host.s3.amazonaws.com/myshellttsbase/examples/zh/ZH/speed_1.0/sent_008.wav) |
| Japanese              | [Link](https://myshell-public-repo-host.s3.amazonaws.com/myshellttsbase/examples/jp/JP/speed_1.0/sent_000.wav) |
| Korean                | [Link](https://myshell-public-repo-host.s3.amazonaws.com/myshellttsbase/examples/kr/KR/speed_1.0/sent_000.wav) |

Some other features include:
- The Chinese speaker supports `mixed Chinese and English`.
- Fast enough for `CPU real-time inference`.

## Usage
- [Use without Installation](docs/quick_use.md)
- [Install and Use Locally](docs/install.md)
- [Training on Custom Dataset](docs/training.md)

The Python API and model cards can be found in [this repo](https://github.com/myshell-ai/MeloTTS/blob/main/docs/install.md#python-api) or on [HuggingFace](https://huggingface.co/myshell-ai).

**Contributing**

If you find this work useful, please consider contributing to this repo.

- Many thanks to [@fakerybakery](https://github.com/fakerybakery) for adding the Web UI and CLI part.

## Authors

- [Wenliang Zhao](https://wl-zhao.github.io) at Tsinghua University
- [Xumin Yu](https://yuxumin.github.io) at Tsinghua University
- [Zengyi Qin](https://www.qinzy.tech) (project lead) at MIT and MyShell

**Citation**
```
@software{zhao2024melo,
  author={Zhao, Wenliang and Yu, Xumin and Qin, Zengyi},
  title = {MeloTTS: High-quality Multi-lingual Multi-accent Text-to-Speech},
  url = {https://github.com/myshell-ai/MeloTTS},
  year = {2023}
}
```

## License

This library is under MIT License, which means it is free for both commercial and non-commercial use.

## Acknowledgements

This implementation is based on [TTS](https://github.com/coqui-ai/TTS), [VITS](https://github.com/jaywalnut310/vits), [VITS2](https://github.com/daniilrobnikov/vits2) and [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2). We appreciate their awesome work.
