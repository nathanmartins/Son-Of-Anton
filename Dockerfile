FROM tensorflow/tensorflow:2.0.0-py3-jupyter

RUN pip3 install --upgrade pip

# OpenCV needs this
RUN apt-get install -y libsm6 libxext6 libxrender-dev

RUN mkdir /usr/src/app

ADD requirements.txt  /usr/src/app
WORKDIR /usr/src/app

RUN pip3 install -r requirements.txt

ADD . /usr/src/app


CMD ["python3", "/usr/src/app/son.py"]
#CMD ["bash", "-c", "source /etc/bash.bashrc && jupyter notebook --notebook-dir=/tf --ip 0.0.0.0 --no-browser --allow-root"]

