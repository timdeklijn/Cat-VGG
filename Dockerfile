FROM tensorflow/tensorflow:2.4.1-gpu
# FROM tensorflow/tensorflow:2.4.1

ADD requirements.txt ./requirements.txt

RUN pip3 install -r requirements.txt --no-cache-dir

ADD vgg ./vgg

ENTRYPOINT ["python3", "-u", "vgg/run.py"]

