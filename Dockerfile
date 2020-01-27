FROM codalab/codalab-legacy:py3

RUN apt-get update -qy
RUN apt-get install -qy docker.io
RUN pip3 install textworld==1.1.1 docker matplotlib
