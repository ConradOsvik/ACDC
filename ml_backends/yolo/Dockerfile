FROM ultralytics/ultralytics:latest

WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive

RUN pip install --no-cache-dir label-studio-ml label-studio-converter redis rq requests git+https://github.com/ultralytics/CLIP.git

COPY model.py .

RUN label-studio-ml init yolo-backend --script model.py

EXPOSE 9090

CMD ["label-studio-ml", "start", "yolo-backend", "--host", "0.0.0.0", "--port", "9090"]
