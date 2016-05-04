#from	ubuntu:14.04
from spark-local

MAINTAINER elliot.ickovic@soteradefense.com

EXPOSE 4040

run	apt-get update
run	apt-get install -y -q wget curl
run	apt-get install -y -q build-essential
run	apt-get install -y -q cmake
run	apt-get install -y -q python2.7 python2.7-dev python-pip
run	wget 'https://pypi.python.org/packages/2.7/s/setuptools/setuptools-0.6c11-py2.7.egg' && /bin/sh setuptools-0.6c11-py2.7.egg && rm -f setuptools-0.6c11-py2.7.egg
#run	curl 'https://raw.github.com/pypa/pip/master/contrib/get-pip.py' | python2.7
run	pip install numpy
run	apt-get install -y -q libavformat-dev libavcodec-dev libavfilter-dev libswscale-dev
run	apt-get install -y -q libjpeg-dev libpng-dev libtiff-dev libjasper-dev zlib1g-dev libopenexr-dev libxine-dev libeigen3-dev libtbb-dev

#OCR:
copy . /srv/software/pst-extraction/
run sudo apt-get update && apt-get install -qy $(cat /srv/software/pst-extraction/apt_get_deps.txt)
run sudo pip install --no-use-wheel --upgrade pip
run sudo apt-get install -y -q cython3
run sudo pip install --upgrade --no-use-wheel -r /srv/software/pst-extraction/pip_get_deps.txt

add	build_opencv.sh	/build_opencv.sh
run	/bin/sh /build_opencv.sh
run	rm -rf /build_opencv.sh
workdir /srv/software/pst-extraction/
#workdir /
cmd ["spark-submit"]

