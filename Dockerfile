FROM python:3.13-trixie
LABEL authors="theo jolivel"

ENV SETUPTOOLS_SCM_PRETEND_VERSION_FOR_MOSAIC_CATEGORIZER=0.0.1

WORKDIR /build
COPY pyproject.toml .
COPY mosaic ./mosaic

#Install dependencies
RUN apt install autoconf automake make gcc bzip2

#Install darshan-utils
RUN wget https://github.com/darshan-hpc/darshan/releases/download/3.5.0/darshan-3.5.0.tar.gz &&\
    tar -xvzf darshan-3.5.0.tar.gz &&\
    cd darshan-3.5.0/darshan-util &&\
    ./configure &&\
    make &&\
    make install

#Patch FTIO
#RUN sed -i 's!(t_step < t\[i + 1\]) or i == n - 1!i == n - 1 or (t_step < t[i + 1])!g' /usr/local/lib/python3.12/site-packages/ftio/freq/discretize.py

RUN pip install --no-cache-dir --upgrade pip setuptools wheel &&\
    pip install --no-cache-dir .

WORKDIR /app

RUN rm -rf /build

ENTRYPOINT ["python"]