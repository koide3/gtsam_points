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
    wget nano build-essential \
    git cmake libeigen3-dev ca-certificates \
    libtbb-dev libboost-all-dev libgtest-dev libmetis-dev \
    libgtsam-dev libgtsam-unstable-dev \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*


COPY . /root/gtsam_ext
WORKDIR /root/gtsam_ext/build
RUN cmake .. \
  -DBUILD_TEST=ON \
  -DBUILD_WITH_CUDA=OFF \
  -DCMAKE_BUILD_TYPE=Release
RUN make -j$(nproc)

WORKDIR /root/gtsam_ext
RUN ./build/gtsam_ext_test

CMD ["bash"]
