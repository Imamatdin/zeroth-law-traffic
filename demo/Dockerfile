FROM python:3.11-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
WORKDIR /app
COPY demo/requirements.txt /app/demo/requirements.txt
RUN pip install --no-cache-dir -r demo/requirements.txt
COPY --chown=user:user src /app/src
COPY --chown=user:user configs /app/configs
COPY --chown=user:user solution.py evaluate.py /app/
COPY --chown=user:user demo /app/demo
RUN python demo/download_weights.py && chown -R user:user /app/weights
USER user
ENV HOME=/home/user PYTHONUNBUFFERED=1 YOLO_AUTOINSTALL=false YOLO_OFFLINE=true CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MPLCONFIGDIR=/tmp/matplotlib
EXPOSE 7860
CMD ["uvicorn", "demo.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
