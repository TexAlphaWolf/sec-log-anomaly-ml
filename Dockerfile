FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN useradd --create-home --uid 10001 loganomaly
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

# Logs go in at /logs (read-only is fine), results come out at /work.
RUN mkdir /logs /work && chown loganomaly /work
USER loganomaly
WORKDIR /work

ENTRYPOINT ["loganomaly"]
CMD ["--help"]
