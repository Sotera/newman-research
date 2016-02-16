FROM houqp/py26:0.0.1

# RUN apt-get purge -y libtiff5
RUN apt-get install -y libjpeg-turbo8 libjbig0 libjpeg8 libtiff5

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app
#COPY requirements.txt /usr/src/app/
#RUN pip install --no-cache-dir -r requirements.txt
COPY . /usr/src/app
# RUN dpkg -i ./libtiff4_3.9.7-2ubuntu1_amd64.deb

#ENV PATH /usr/src/app
CMD ["python", "/usr/src/app/transaction_extraction_ocr_deconvolution.py"]


#Previous attempted docker builds:
#---------------------------------
#FROM python:2.7
#RUN mkdir -p /usr/src/app
#WORKDIR /usr/src/app
#COPY requirements.txt /usr/src/app/
#RUN pip install --no-cache-dir -r requirements.txt
#COPY . /usr/src/app
#RUN export LD_LIBRARY_PATH=/usr/src/app
#RUN ldconfig
#RUN apt-get install -qy python2.7
#ENV PATH /usr/src/app
#CMD ["python", "/usr/src/app/transaction_extraction_ocr_deconvolution.py"]

#FROM stackbrew/debian:jessie
#RUN apt-get install -qy python2
#RUN apt-get install -qy python2-pip
#ADD requirements.txt /
#RUN pip-3.3 install -r requirements.txt
#COPY . /src
#CMD ["python", "/src/transaction_extraction_ocr_deconvolution.py"]

#DOCKER-VERSION 1.9.1
#RUN pip install image
#FROM python:2
#WORKDIR /srv
#ADD ./requirements.txt /srv/requirements.txt
#RUN pip install -r requirements.txt
#ADD . /srv
#RUN python setup.py install
#COPY . /src
