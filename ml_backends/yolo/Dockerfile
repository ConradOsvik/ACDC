FROM ultralytics/ultralytics:latest

WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive

COPY yolo/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY utils.py .
COPY yolo/model.py .

RUN label-studio-ml init yolo-backend --script model.py && cp utils.py yolo-backend/

EXPOSE 9090

CMD ["label-studio-ml", "start", "yolo-backend", "--host", "0.0.0.0", "--port", "9090"]
