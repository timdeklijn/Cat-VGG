FROM python:3.9

RUN pip install mlflow==1.17

EXPOSE 5000/tcp

COPY tracking_server/run.sh .

ENTRYPOINT [ "mlflow", "server", "--host", "0.0.0.0"]
CMD [ "--backend-store-uri", "/mlflow", \
      "--default-artifact-root", "file:///artefacts"]
