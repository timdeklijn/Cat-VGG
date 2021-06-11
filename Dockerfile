# FROM tensorflow/tensorflow:2.5.0-gpu
FROM tensorflow/tensorflow:2.4.1

ADD requirements.txt ./requirements.txt

RUN pip3 install -r requirements.txt --no-cache-dir
RUN pip3 install matplotlib==3.3.4 --no-cache-dir

ADD vgg ./vgg

ENTRYPOINT ["python3", "-u", "vgg/run.py"]

