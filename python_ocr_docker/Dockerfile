FROM houqp/py26:0.0.1
RUN apt-get install -y libjpeg-turbo8 libjbig0 libjpeg8 libtiff5

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app
#COPY requirements.txt /usr/src/app/
#RUN pip install --no-cache-dir -r requirements.txt
COPY . /usr/src/app

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
RUN sudo pip install Pillow

RUN sudo pip install numpy
RUN sudo apt-get update && apt-get install -qy \
    	 	 	     python-skimage \
    	 	 	     libblas-dev \
			     liblapack-dev \
			     libatlas-base-dev \
			     gfortran \
			     tesseract-ocr \
			     emacs
RUN sudo pip install scipy
CMD ["python", "/usr/src/app/transaction_extraction_ocr_deconvolution.py"]
