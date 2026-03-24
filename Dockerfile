FROM dustynv/l4t-pytorch:r36.4.0

ENV PIP_INDEX_URL=https://pypi.jetson-ai-lab.io/jp6/cu126

# Upgrade to torch 2.8 (base has 2.4, TRL needs FSDPModule from 2.6+)
RUN pip install --no-cache-dir 'torch==2.8.0' 'torchvision==0.23.0' 'numpy<2'

# Save torch 2.8 so we can restore after other installs
RUN cp -r /usr/local/lib/python3.10/dist-packages/torch /tmp/torch_backup && \
    cp -r /usr/local/lib/python3.10/dist-packages/torchvision /tmp/torchvision_backup

# Install ML packages with --no-deps (prevents overwriting Jetson torch)
RUN pip install --no-cache-dir --no-deps \
    bitsandbytes xformers peft trl transformers accelerate datasets \
    safetensors huggingface_hub wandb matplotlib pandas sentencepiece \
    unsloth_zoo nest-asyncio cut_cross_entropy msgspec triton \
    'unsloth @ git+https://github.com/unslothai/unsloth.git'

# Install pure-python sub-deps
RUN pip install --no-cache-dir --no-deps \
    hf_transfer hf-xet regex tokenizers multiprocess dill xxhash \
    pyarrow fsspec filelock packaging tqdm pyyaml requests jinja2 \
    psutil docstring-parser tyro rich typer shellingham typeguard \
    annotated-doc anyio httpx httpcore h11 sniffio certifi idna \
    urllib3 charset-normalizer markupsafe mpmath sympy networkx \
    aiohttp aiosignal frozenlist multidict yarl propcache attrs \
    async-timeout aiohappyeyeballs exceptiongroup click pyparsing \
    cycler contourpy fonttools kiwisolver python-dateutil six pytz tzdata \
    'antlr4-python3-runtime==4.9.3' sentry-sdk setproctitle gitpython gitdb smmap

# Fix pyparsing: system version too old for matplotlib
RUN pip install --no-cache-dir --force-reinstall pyparsing && \
    rm -f /usr/lib/python3/dist-packages/pyparsing.py

# Restore Jetson torch 2.8 (pip deps may have overwritten it)
RUN rm -rf /usr/local/lib/python3.10/dist-packages/torch && \
    cp -r /tmp/torch_backup /usr/local/lib/python3.10/dist-packages/torch && \
    rm -rf /usr/local/lib/python3.10/dist-packages/torchvision && \
    cp -r /tmp/torchvision_backup /usr/local/lib/python3.10/dist-packages/torchvision && \
    rm -rf /tmp/torch_backup /tmp/torchvision_backup

WORKDIR /workspace
