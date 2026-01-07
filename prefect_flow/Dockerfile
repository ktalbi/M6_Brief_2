FROM prefecthq/prefect:2-latest

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
 && rm -rf /var/lib/apt/lists/*


COPY prefect_flow/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt


COPY prefect_flow/flow.py /app/flow.py


COPY fastapi_app/app /app/app

RUN test -f /app/app/__init__.py || touch /app/app/__init__.py \
 && test -f /app/app/ml/__init__.py || true

ENV PYTHONPATH=/app

CMD ["python", "/app/flow.py"]
