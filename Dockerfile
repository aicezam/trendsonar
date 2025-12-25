FROM public.ecr.aws/docker/library/python:3.11-slim

WORKDIR /app

# 设置时区为上海
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 安装基础构建依赖
# 替换系统源为阿里云 (加速 apt install)
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources

# build-essential: 编译 Python C 扩展
# curl: 常用工具
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

# --- 安装 Playwright 浏览器环境 (Crawl4AI 需要) ---
# 1. 安装 Chromium 及其系统依赖
# 2. 清理缓存以减小镜像体积
# 注意：这将显著增加镜像大小
# 设置 Playwright 下载镜像 (加速浏览器下载)
ENV PLAYWRIGHT_DOWNLOAD_HOST="https://npmmirror.com/mirrors/playwright/"
RUN playwright install --with-deps chromium
# ----------------------------------------------------

COPY . .

# 暴露端口 (与 config.py 中默认一致)
EXPOSE 8193

CMD ["python", "main.py"]
