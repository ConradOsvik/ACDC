FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model.py _wsgi.py ./

EXPOSE 9090

ENTRYPOINT ["label-studio-ml", "start", "./", "--port", "9090"]
