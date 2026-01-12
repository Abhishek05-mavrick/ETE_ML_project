FROM python:3.11-slim
WORKDIR /End-to-End

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential gcc g++ python3-dev libgomp1 libssl-dev libffi-dev && \
    rm -rf /var/lib/apt/lists/*

# ensure pip/build tools are recent
RUN pip install --upgrade pip setuptools wheel

COPY requirements.txt ./
RUN  pip install -r requirements.txt

COPY ./ ./
EXPOSE 5000

CMD ["python","app.py"]