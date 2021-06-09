# FROM tensorflow/tensorflow:2.5.0-gpu
FROM tensorflow/tensorflow:2.5.0

ADD requirements.txt ./requirements.txt

RUN pip3 install -r requirements.txt --no-cache-dir
RUN pip3 install matplotlib==3.3.4 --no-cache-dir

ADD vgg ./vgg

CMD ["python3", "./vgg/run.py"]
