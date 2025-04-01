FROM nvidia/cuda:11.7.1-runtime-ubuntu20.04 as builder

WORKDIR /df_detect

COPY ./requirement.txt .

ENV TZ=Asia/Shanghai

RUN apt-get update -qq && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends tzdata && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    apt-get install -y --no-install-recommends python3 build-essential libgl1-mesa-dev cmake python3-pip libopencv-dev python3-dev python3-venv curl && \
    python3 -m venv /venv && \
    /venv/bin/pip install --upgrade pip setuptools wheel && \
    /venv/bin/pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117 && \
    /venv/bin/pip install --no-cache-dir -r requirement.txt -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    cd / && tar -czvf /venv.tar.gz /venv && \
    rm -rf /var/lib/apt/lists/*

 # Second stage: create the runtime image
FROM nvidia/cuda:11.7.1-runtime-ubuntu20.04

WORKDIR /df_detect

COPY --from=builder /venv.tar.gz /venv.tar.gz
COPY ./start.sh /

COPY ./detect_from_video.py ./
COPY ./detect_from_image.py ./
COPY ./detect_from_audio.py ./
COPY ./lfcc.py ./
COPY ./models/ ./models/
COPY ./dataset/ ./dataset/

RUN apt-get update
RUN apt-get install -y --no-install-recommends python3 libx11-6 libpng16-16 libjpeg8
RUN rm -rf /var/lib/apt/lists/*

# Make sure scripts in .local are usable:
ENV PATH=/venv/bin:$PATH
ENTRYPOINT ["bash", "/start.sh"]