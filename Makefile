PROJECT_NAME=deepmind_tracking
VERSION=0.0.1

IMAGE_NAME=$(PROJECT_NAME):$(VERSION)
CONTAINER_NAME=--name=$(PROJECT_NAME)

GPUS=--gpus=all  # Specifies which GPUs the container can see
NET=--net=host
IPC=--ipc=host
BUILD_NET=--network=host

.PHONY: all build stop run logs

all: build stop run logs

build:
	docker build $(BUILD_NET) -t $(IMAGE_NAME) -f Dockerfile .

stop:
	docker stop $(shell docker container ls -q --filter name=$(PROJECT_NAME)*)

kill:
	docker kill $(shell docker container ls -q --filter name=$(PROJECT_NAME)*)
	docker rm $(shell docker container ls -q --filter name=$(PROJECT_NAME)*)

run:
	docker run --rm -it $(GPUS) \
		$(NET) $(IPC) \
		-v $(shell pwd):/workdir/ \
		$(CONTAINER_NAME) \
		$(IMAGE_NAME) \
		bash

logs:
	docker logs -f $(PROJECT_NAME)