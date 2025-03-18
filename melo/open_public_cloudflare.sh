#!/bin/bash

# 스크립트 이름: publish_tensorboard.sh

# 변수 설정
WORKSPACE_DIR="/home/ydk/workspace/melo-tts/melo"
LOG_DIR="logs/default_5"
TENSORBOARD_PORT=6006
TENSORBOARD_LOG="tensorboard.log"
CLOUDFLARE_LOG="cloudflare.log"

# 서비스 중지 함수
stop_services() {
  echo "TensorBoard와 Cloudflare Tunnel 종료 중..."
  pkill -f tensorboard
  pkill -f cloudflared
  echo "모든 서비스가 중지되었습니다."
  exit 0
}

# cloudflared 설치 확인 및 설치 함수
check_and_install_cloudflared() {
  if ! command -v cloudflared &> /dev/null; then
    echo "cloudflared가 설치되어 있지 않습니다. 설치를 시작합니다..."
    
    # 아키텍처 확인
    ARCH=$(uname -m)
    if [[ "$ARCH" == "x86_64" ]]; then
      CLOUDFLARED_ARCH="amd64"
    elif [[ "$ARCH" == "aarch64" || "$ARCH" == "arm64" ]]; then
      CLOUDFLARED_ARCH="arm64"
    else
      echo "지원되지 않는 아키텍처: $ARCH"
      exit 1
    fi
    
    # 최신 버전 다운로드
    echo "Cloudflared 다운로드 중..."
    curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-$CLOUDFLARED_ARCH -o cloudflared
    
    # 실행 권한 부여
    chmod +x cloudflared
    
    # 설치
    echo "Cloudflared 설치 중..."
    sudo mv cloudflared /usr/local/bin/
    
    # 설치 확인
    if command -v cloudflared &> /dev/null; then
      echo "Cloudflared가 성공적으로 설치되었습니다."
    else
      echo "Cloudflared 설치에 실패했습니다."
      exit 1
    fi
  else
    echo "Cloudflared가 이미 설치되어 있습니다."
  fi
}

# 종료 명령 확인
if [ "$1" = "stop" ]; then
  stop_services
fi

# Cloudflared 설치 확인 및 필요시 설치
check_and_install_cloudflared

# 이미 실행 중인 프로세스 종료
echo "기존 프로세스 종료 중..."
pkill -f tensorboard
pkill -f cloudflared
sleep 2

# 작업 디렉토리로 이동
cd $WORKSPACE_DIR

# TensorBoard 시작
echo "TensorBoard 시작 중..."
nohup uv run tensorboard --logdir $LOG_DIR --port $TENSORBOARD_PORT --bind_all > $TENSORBOARD_LOG 2>&1 &
TENSORBOARD_PID=$!
echo "TensorBoard가 PID $TENSORBOARD_PID로 시작되었습니다."

# 잠시 대기 (TensorBoard가 시작될 때까지)
sleep 3

# TensorBoard 시작 확인
if ! ps -p $TENSORBOARD_PID > /dev/null; then
  echo "TensorBoard 시작에 실패했습니다. 로그를 확인하세요: $TENSORBOARD_LOG"
  exit 1
fi

# Cloudflare Tunnel 시작
echo "Cloudflare Tunnel 시작 중..."
nohup cloudflared tunnel --url http://localhost:$TENSORBOARD_PORT > $CLOUDFLARE_LOG 2>&1 &
CLOUDFLARE_PID=$!
echo "Cloudflare Tunnel이 PID $CLOUDFLARE_PID로 시작되었습니다."

# 생성된 URL 확인을 위해 잠시 대기
echo "URL 생성 중..." 
for i in {1..10}; do
  sleep 1
  echo -n "."
  if grep -q "Your quick Tunnel has been created" $CLOUDFLARE_LOG; then
    echo ""
    break
  fi
  
  if [ $i -eq 10 ]; then
    echo ""
    echo "시간 초과: URL 생성을 확인할 수 없습니다. 로그를 확인하세요: $CLOUDFLARE_LOG"
  fi
done

# 생성된 URL 표시
echo "TensorBoard 공개 URL:"
grep "Your quick Tunnel has been created" $CLOUDFLARE_LOG | grep -o "https://[^ ]*" || echo "URL을 찾을 수 없습니다. 로그를 확인하세요: $CLOUDFLARE_LOG"

echo ""
echo "로그 확인:"
echo "TensorBoard 로그: tail -f $TENSORBOARD_LOG"
echo "Cloudflare 로그: tail -f $CLOUDFLARE_LOG"

echo ""
echo "프로세스 확인:"
ps aux | grep tensorboard | grep -v grep
ps aux | grep cloudflared | grep -v grep

echo ""
echo "종료하려면: bash $(basename $0) stop"