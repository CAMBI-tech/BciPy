FROM ubuntu:14.04
MAINTAINER Shiran Dudy <dudy@ohsu.edu>
RUN apt-get update
RUN apt-get install -y wget
RUN apt-get install -y build-essential
RUN DEBIAN_FRONTEND=noniteractive apt-get install -y -q python-dev python-pip
RUN apt-get install -y zlib1g-dev
ENV Version openfst-1.6.3
ENV LD_LIBRARY_PATH /usr/local/lib
RUN wget http://openfst.org/twiki/pub/FST/FstDownload/${Version}.tar.gz
RUN tar -xzvf ${Version}.tar.gz
WORKDIR /${Version}
RUN ./configure --enable-far --enable-python --enable-lookahead-fsts --enable-special
RUN make
RUN make install
WORKDIR ../
ADD ./lm/requirements.txt /tmp/requirements.txt
RUN pip install -qr /tmp/requirements.txt
ADD ./lm /opt/lm/
ADD ./ebitweight /tmp/ebitweight
WORKDIR /tmp/ebitweight
RUN python setup.py build_ext --inplace
WORKDIR ../
ADD ./ebitweight/ebitweight.pyx /opt/lm/ebitweight.pyx
RUN mv ./ebitweight/ebitweight.so /opt/lm/ebitweight.so
RUN rm -rf ./ebitweight
RUN rm -rf /tmp/ebitweight
RUN rm -rf ./requirements.txt
ADD ./specializer /tmp/specializer
WORKDIR /tmp/specializer
ENV LD_LIBRARY_PATH "$LD_LIBRARY_PATH:/usr/local/lib/fst"
RUN echo ${Version}
RUN cp /${Version}/src/extensions/special/.libs/rho-fst.so /tmp/specializer/rho-fst.so
RUN cp /${Version}/src/extensions/special/.libs/phi-fst.so /tmp/specializer/phi-fst.so
RUN cp /${Version}/src/extensions/special/.libs/sigma-fst.so /tmp/specializer/sigma-fst.so
RUN cp /${Version}/src/extensions/python/pywrapfst.pxd /tmp/specializer/pywrapfst.pxd
RUN cp /${Version}/src/extensions/python/pywrapfst.pyx /tmp/specializer/pywrapfst.pyx
ADD ./specialfiles/ios.pxd /tmp/specializer/ios.pxd
ADD ./specialfiles/basictypes.pxd /tmp/specializer/basictypes.pxd
ADD ./specialfiles/fst.pxd /tmp/specializer/fst.pxd
ADD ./specialfiles/memory.pxd /tmp/specializer/memory.pxd
ADD ./specialfiles/fst_util.pxd /tmp/specializer/fst_util.pxd
RUN python setup.py build_ext --inplace
WORKDIR unitTest
RUN ./test_specializer.sh
WORKDIR ../../
ADD ./specializer/specializer.pyx /opt/lm/specializer.pyx
RUN mv ./specializer/specializer.so /opt/lm/specializer.so
RUN rm -rf /tmp/specializer
RUN rm /${Version}.tar.gz
RUN rm -rf /${Version}
RUN pip install openfst
WORKDIR /opt/lm
EXPOSE 5000
