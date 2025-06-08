# Makefile for MLOps Video Processing Pipeline

# Configuration
IMAGE_NAME = video-pipeline
VERSION = latest
INPUT_DIR = ./input
OUTPUT_DIR = ./output
CONFIG_FILE = default_model_config.yaml

.PHONY: all build run test lint clean help

all: build run

## Build Docker image
build:
	@echo "Building Docker image..."
	docker build -t $(IMAGE_NAME):$(VERSION) .

## Run pipeline with sample input
run:
	@echo "Running pipeline..."
	docker run -it --rm \
		-v $(INPUT_DIR):/input \
		-v $(OUTPUT_DIR):/output \
		$(IMAGE_NAME):$(VERSION) \
		--input-video /input/sample.mp4 \
		--output-dir /output \
		--model-config /app/$(CONFIG_FILE)

## Run unit tests
test:
	@echo "Running tests..."
	docker build -t $(IMAGE_NAME)-test .
	docker run $(IMAGE_NAME)-test pytest tests/ -v

## Lint code
lint:
	@echo "Linting code..."
	flake8 src/ --max-line-length=120
	black --check src/

## Clean generated files
clean:
	@echo "Cleaning up..."
	rm -rf $(OUTPUT_DIR)/*
	docker rmi $(IMAGE_NAME):$(VERSION) || true

## Push image to registry (example for AWS ECR)
push:
	$(eval ACCOUNT_ID := $(shell aws sts get-caller-identity --query Account --output text))
	$(eval REGION := $(shell aws configure get region))
	aws ecr get-login-password | docker login --username AWS --password-stdin $(ACCOUNT_ID).dkr.ecr.$(REGION).amazonaws.com
	docker tag $(IMAGE_NAME):$(VERSION) $(ACCOUNT_ID).dkr.ecr.$(REGION).amazonaws.com/$(IMAGE_NAME):$(VERSION)
	docker push $(ACCOUNT_ID).dkr.ecr.$(REGION).amazonaws.com/$(IMAGE_NAME):$(VERSION)

## Show help
help:
	@echo "Available targets:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
