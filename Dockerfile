FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

RUN apt-get update && apt-get install -y \
    build-essential \
    openmpi-bin \
    libopenmpi-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app 

RUN mkdir -p data

RUN make clean && make

CMD ./GeneradorPuntosNDimensiones && mpirun -mca btl_vader_single_copy_mechanism none -np 4 ./main_mpi