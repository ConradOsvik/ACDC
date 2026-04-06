FROM ultralytics/ultralytics:latest

WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive

RUN pip install --no-cache-dir \
    label-studio-ml label-studio-converter redis rq requests \
    timm huggingface-hub safetensors \
    git+https://github.com/ultralytics/CLIP.git

COPY utils.py .
COPY sam3/model.py .

RUN label-studio-ml init sam3-backend --script model.py && cp utils.py sam3-backend/

EXPOSE 9090

CMD ["label-studio-ml", "start", "sam3-backend", "--host", "0.0.0.0", "--port", "9090"]
