FROM python:3.10-slim

# Install Rust, Cargo, and build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && curl https://sh.rustup.rs -sSf | sh -s -- -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Add Cargo to PATH
ENV PATH="/root/.cargo/bin:${PATH}"

# Install maturin and Python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir maturin langchain_community python-dotenv openai langchain-huggingface faiss-cpu

CMD ["bash"]
#CMD ["python", "app/test.py"]