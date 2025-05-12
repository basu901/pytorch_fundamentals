FROM python:3.10-slim

ARG MLFLOW_ARTIFACT_STORE
ENV PYTHONBUFFERED=1 \
    VIRTUAL_ENV="${HOME}/venv" \
    PATH="${HOME}/venv/bin:${PATH}" \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8 \
    DEBIAN_FRONTEND='noninteractive' \
    BUILD_POETRY_LOCK="${HOME}/poetry.lock.build"

RUN apt-get -qq update \
    && apt-get -qq -y install git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get -qq -y clean


RUN mkdir -p "${MLFLOW_ARTIFACT_STORE}" /app

COPY ./docker/run-server.sh /
RUN chmod +x /run-server.sh

RUN HOME=/tmp pip install --no-cache-dir poetry==1.5.1

COPY ./pyproject.toml ./*.lock /app/

WORKDIR /app
RUN python3.10 -m venv "${VIRTUAL_ENV}" \
    && pip install --upgrade pip \
    && poetry install \
    && cp poetry.lock "${BUILD_POETRY_LOCK}" \
    && rm -rf "${HOME}/.cache/*"

CMD ["/run-server.sh"]