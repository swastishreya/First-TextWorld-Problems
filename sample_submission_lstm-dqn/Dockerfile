FROM tavianator/textworld-codalab

MAINTAINER TextWorld Team <textworld@microsoft.com>

RUN apt-get update -qy
RUN apt-get install -qy python3-pip wget graphviz
RUN pip3 install spacy
RUN python3 -m spacy download en
RUN pip3 install torchvision
