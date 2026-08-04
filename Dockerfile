# 自己評価キット用 Dockerfile（本番採点環境と同じベース構成）
#
#   docker build -t parc2026 .
#
#   # 対話シェル（中で README の自己評価コマンドをそのまま使える）
#   docker run -it --rm parc2026
#
#   # 提出 zip をエンドツーエンド評価
#   docker run --rm -v $PWD/my_submission.zip:/sub.zip parc2026 \
#       python evaluate.py /sub.zip --n-episodes 2
#
# 環境構築は setup.sh（ローカル構築と同一手順）をビルド時に実行する。
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV MUJOCO_GL=osmesa
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip \
    libosmesa6 libosmesa6-dev \
    libgl1 libglfw3 libglew-dev \
    libegl1 \
    libsm6 libxext6 libxrender-dev \
    libglib2.0-0 \
    libmagickwand-dev \
    build-essential cmake git wget curl zip unzip ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY . /workspace

# venv 作成・依存・LIBERO-plus 取得とパッチ・アセット・config.yaml まで一式
RUN bash setup.sh

ENV PATH="/workspace/venv/bin:${PATH}"
ENV PYTHONPATH="/workspace/LIBERO-plus:/workspace:/workspace/compe"
ENV LIBERO_ROOT="/workspace/LIBERO-plus"

CMD ["bash"]
