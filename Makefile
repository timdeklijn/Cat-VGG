.PHONY: tracking_server

tracking_server:
	mlflow server \
	    --backend-store-uri ./mlflow \
	    --default-artifact-root ./mlflow \
	    --host 127.0.0.1

docker_build:
	docker build -t vgg:latest .

docker_run:
	docker run --rm -v $(shell PWD)/data:/data --network="host" vgg:latest
