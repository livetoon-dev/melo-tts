from huggingface_hub import snapshot_download

# 사용자 지정 경로에 모델 다운로드
model_path = "/home/ydk/workspace/melo-tts/melo/pretrain/melotts-pretrain-beta"
snapshot_download(repo_id="LiveTaro/melotts-pretrain-beta", local_dir=model_path)
