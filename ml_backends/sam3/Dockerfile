FROM ultralytics/ultralytics:latest

WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive

COPY sam3/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY utils.py .
COPY sam3/model.py .

RUN label-studio-ml init sam3-backend --script model.py && cp utils.py sam3-backend/

EXPOSE 9090

CMD ["label-studio-ml", "start", "sam3-backend", "--host", "0.0.0.0", "--port", "9090"]
