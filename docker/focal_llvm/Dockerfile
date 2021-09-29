FROM ubuntu:focal

ENV DEBIAN_FRONTEND=noninteractive
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

RUN apt-get update \
  && apt-get upgrade -y \
  # add gtsam respository
  && apt-get install --no-install-recommends -y \
    software-properties-common \
  && add-apt-repository ppa:borglab/gtsam-develop \
  # install dependencies
  && apt-get install --no-install-recommends -y \
    wget nano build-essential clang lld \
    git cmake libeigen3-dev ca-certificates \
    libtbb-dev libboost-all-dev libgtest-dev libmetis-dev \
    libpthread-stubs0-dev libgtsam-dev libgtsam-unstable-dev \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/ld ld /usr/bin/ld.lld 50

COPY . /root/gtsam_ext
WORKDIR /root/gtsam_ext/build
RUN CC=clang CXX=clang++ \
  cmake .. \
  -DBUILD_TESTS=ON \
  -DBUILD_WITH_CUDA=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_WITH_SYSTEM_EIGEN=OFF
RUN make -j$(nproc)
RUN make test

CMD ["bash"]
