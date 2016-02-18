FROM houqp/py26:0.0.1
RUN apt-get install -y libjpeg-turbo8 libjbig0 libjpeg8 libtiff5

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app
#COPY requirements.txt /usr/src/app/
#RUN pip install --no-cache-dir -r requirements.txt
COPY . /usr/src/app

RUN sudo apt-get install -qy wget
#RUN wget --no-check-certificate http://pypi.python.org/packages/source/d/distribute/distribute-0.6.35.tar.gz
#RUN tar xf distribute-0.6.35.tar.gz

RUN sudo apt-get update && apt-get install -qy python2.7-dev
#RUN cd distribute-0.6.35
#RUN python2.7 /usr/src/app/distribute-0.6.35/setup.py install
RUN wget https://bootstrap.pypa.io/get-pip.py
RUN sudo python2.7 get-pip.py

RUN sudo pip2.7 install numpy
RUN sudo pip2.7 install cython
RUN sudo apt-get update && apt-get install -qy \
    	 	 	python2.7-dev \
    	    	    	libpython-dev \
			python-dev \
			libjpeg-dev \
			libjpeg8-dev \
			libpng3 \
			libfreetype6-dev

RUN ln -s /usr/lib/i386-linux-gnu/libfreetype.so /usr/lib
RUN ln -s /usr/lib/i386-linux-gnu/libjpeg.so /usr/lib
RUN ln -s /usr/lib/i386-linux-gnu/libz.so /usr/lib
RUN sudo pip2.7 install Pillow

RUN sudo apt-get update && apt-get install -qy \
    	 	 	     libblas-dev \
			     liblapack-dev \
			     libatlas-base-dev \
			     gfortran \
			     tesseract-ocr \
			     emacs
RUN sudo pip2.7 install --no-cache-dir -r requirements.txt
RUN sudo apt-get update && apt-get install -qy python-skimage
RUN sudo pip2.7 install scipy
RUN sudo pip2.7 install pytesseract
#RUN cp -r /usr/lib/python2.6/site-packages/ /usr/lib/python2.7/dist-packages
CMD ["python", "/usr/src/app/transaction_extraction_ocr_deconvolution.py"]
