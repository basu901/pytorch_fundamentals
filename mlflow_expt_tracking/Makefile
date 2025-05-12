include .envs/.mlflow-common
include .envs/.mlflow-dev
include .envs/.postgres

export

DOCKER_COMPOSE_RUN = docker-compose run --rm mlflow-server
lock-dependencies: BUILD_POETRY_LOCK = /poetry.lock.build

build:
	docker-compose build

up:
	docker-compose up -d

down:
	docker-compose down

exec-in: up
	docker exec -it local-mlflow-tracking-server bash

lock-dependencies:
	${DOCKER_COMPOSE_RUN} bash -c "if [ -e ${BUILD_POETRY_LOCK} ]; then cp ${BUILD_POETRY_LOCK} ./poetry.lock; else poetry lock; fi"