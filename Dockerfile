FROM python:3.12-alpine AS build
LABEL authors="theo jolivel"

WORKDIR /build
COPY requirements.txt .

#Install dependencies
RUN apk --no-cache add git autoconf automake build-base gcc llvm15-dev clang15-dev libedit-dev libtool

#Install darshan-utils
RUN wget https://web.cels.anl.gov/projects/darshan/releases/darshan-3.4.7.tar.gz &&\
    tar -xvzf darshan-3.4.7.tar.gz &&\
    cd darshan-3.4.7 &&\
    ./prepare.sh &&\
    cd darshan-util &&\
    ./configure &&\
    make &&\
    make install

#Install Python packages
ENV LLVM_CONFIG=/usr/bin/llvm-config-15
RUN pip install --no-cache-dir --upgrade pip &&\
    pip install --no-cache-dir -r requirements.txt

#Install FTIO
RUN git clone https://github.com/tuda-parallel/FTIO.git &&\
    cd FTIO &&\
    git checkout development &&\
    git reset --hard c64a45bbad86e6a9d040fbb5eadd5be7efa3fa87 &&\
    pip install --no-cache-dir '.[external-libs]'

#Patch FTIO
RUN sed -i "316i\\        # Debug flags\n        parser.add_argument('--name_debug', '--name_debug', dest='name_debug', type = str, help = 'Name of the trace for debugging purposes')\n        parser.set_defaults(name_debug = '')" /usr/local/lib/python3.12/site-packages/ftio/parse/args.py &&\
    sed -i "7i import sys" /usr/local/lib/python3.12/site-packages/ftio/freq/discretize.py &&\
    sed -i "74i\\            print(f'{args.name_debug}: {memory_limit/1e9:.3} FTIO memory cap reached, sampling frequency:  {freq:.3e} Hz, original: {old_freq:.3e} Hz" /usr/local/lib/python3.12/site-packages/ftio/freq/discretize.py &&\
    sed -i 's!(t_step < t\[i + 1\]) or i == n - 1!(i == n - 1 or (t_step < t[i + 1]))!g' /usr/local/lib/python3.12/site-packages/ftio/freq/discretize.py

FROM python:3.12-alpine

COPY --from=build /usr/local/lib/libdarshan* /usr/local/lib
COPY --from=build /usr/local/lib/python3.12/site-packages/ /usr/local/lib/python3.12/site-packages/

RUN apk add llvm15-dev libgomp

WORKDIR /app

ENTRYPOINT ["python"]