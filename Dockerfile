FROM public.ecr.aws/docker/library/python:3.11-slim

WORKDIR /app

ENV TZ=Asia/Shanghai \
    DEBIAN_FRONTEND=noninteractive \
    # 关键：使用已经验证有对应浏览器版本的镜像
    PLAYWRIGHT_DOWNLOAD_HOST=https://npmmirror.com/mirrors/playwright

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources \
    && apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 固定一个已知有镜像同步的 Playwright 版本
RUN pip install --no-cache-dir playwright==1.56.0 \
    && python -m playwright install-deps chromium \
    && python -m playwright install chromium

COPY . .

EXPOSE 8193

CMD ["python", "main.py"]
