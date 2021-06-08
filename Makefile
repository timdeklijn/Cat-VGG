.PHONY: tracking_server

tracking_server:
	mlflow server \
	    --backend-store-uri ./mlflow \
	    --default-artifact-root ./mlflow \
	    --host 0.0.0.0

