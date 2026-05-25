FROM python:3.11-slim

# Set timezone
ENV TZ=Asia/Shanghai

# Replace Debian mirror with Aliyun mirror for faster downloads in China
RUN sed -i 's|https\?://deb\.debian\.org/debian|https://mirrors.aliyun.com/debian|g' /etc/apt/sources.list.d/debian.sources 2>/dev/null || true && \
    apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/* && apt-get clean

# Install uv securely: download, install to system path, cleanup
COPY build/uv-install.sh /uv-installer.sh
RUN chmod +x /uv-installer.sh && \
    sh /uv-installer.sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv && \
    rm -f /uv-installer.sh && \
    rm -rf /root/.local

# Set working directory
WORKDIR /app

# Copy only dependency definition (not application code) for installation
COPY pyproject.toml ./

# Install project dependencies using uv with Tsinghua PyPI mirror
# Using --system to install to system Python environment in container
RUN --mount=type=cache,target=/root/.cache/uv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_DEV=1 \
    UV_HTTP_TIMEOUT=1200 \
    uv pip install --system --index-url https://pypi.tuna.tsinghua.edu.cn/simple .

# Create non-root user with UID 1008 and set proper ownership
RUN useradd -m -u 1008 -s /bin/bash appuser && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

EXPOSE 8014