ARG BASE_IMAGE=ubuntu:noble
ARG CUDA_BASE_IMAGE=nvidia/cuda:12.5.1-devel-ubuntu24.04

##### gtsam #####
FROM ${BASE_IMAGE} AS gtsam

ENV DEBIAN_FRONTEND=noninteractive
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

RUN apt-get update
RUN apt-get install -y --no-install-recommends
RUN apt-get install -y --no-install-recommends build-essential  \
    git cmake libeigen3-dev ca-certificates libboost-all-dev libmetis-dev file

RUN git clone https://github.com/borglab/gtsam.git /root/gtsam -b 4.2a9
WORKDIR /root/gtsam/build
# Rename libgtsam-dev to libgtsam-notbb-dev
RUN sed -i 's/libgtsam-dev/libgtsam-notbb-dev/g' ../cmake/HandleCPack.cmake
# Build gtsam
RUN cmake .. -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF \
  -DGTSAM_BUILD_TESTS=OFF \
  -DGTSAM_WITH_TBB=OFF \
  -DGTSAM_BUILD_WITH_MARCH_NATIVE=OFF \
  -DGTSAM_USE_SYSTEM_EIGEN=ON \
  -DGTSAM_USE_SYSTEM_METIS=ON \
  -DCPACK_GENERATOR=DEB \
  -DCPACK_DEBIAN_FILE_NAME=DEB-DEFAULT \
  -DCPACK_SET_DESTDIR=ON \
  -DCPACK_DEBIAN_PACKAGE_SHLIBDEPS=ON
RUN make -j$(nproc)
RUN cpack

##### gtsam_points #####
FROM ${BASE_IMAGE} as gtsam_points

ENV DEBIAN_FRONTEND=noninteractive
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

RUN apt-get update
RUN apt-get install -y --no-install-recommends
RUN apt-get install -y --no-install-recommends build-essential  \
    git cmake libeigen3-dev ca-certificates libboost-all-dev libmetis-dev
RUN apt-get install -y --no-install-recommends curl libomp-dev libtbb-dev file

COPY --from=gtsam /root/gtsam/build/*.deb /root/gtsam/build/
RUN apt install -y /root/gtsam/build/*.deb

COPY . /root/gtsam_points
WORKDIR /root/gtsam_points/build
RUN cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCPACK_DEBIAN_PACKAGE_DEPENDS="libgtsam-notbb-dev" \
  -DCPACK_DEBIAN_PACKAGE_NAME="libgtsam-points-dev"
RUN make -j$(nproc)
RUN cpack


##### gtsam_points #####
FROM ${CUDA_BASE_IMAGE} as gtsam_points_cuda

ARG CUDA_LABEL=12.5

ENV DEBIAN_FRONTEND=noninteractive
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

RUN apt-get update
RUN apt-get install -y --no-install-recommends
RUN apt-get install -y --no-install-recommends build-essential  \
    git cmake libeigen3-dev ca-certificates libboost-all-dev libmetis-dev
RUN apt-get install -y --no-install-recommends libomp-dev libtbb-dev file

COPY --from=gtsam /root/gtsam/build/*.deb /root/gtsam/build/
RUN apt install -y /root/gtsam/build/*.deb

COPY . /root/gtsam_points
WORKDIR /root/gtsam_points/build

RUN cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_WITH_CUDA=ON \
  -DBUILD_WITH_CUDA_MULTIARCH=ON \
  -DCPACK_DEBIAN_PACKAGE_DEPENDS="libgtsam-notbb-dev" \
  -DCPACK_DEBIAN_PACKAGE_NAME="libgtsam-points-cuda${CUDA_LABEL}-dev"
RUN make -j$(nproc)
RUN cpack


##### extract #####
FROM ${BASE_IMAGE} AS extract

COPY --from=gtsam /root/gtsam/build/*.deb /root/deb/
COPY --from=gtsam_points /root/gtsam_points/build/*.deb /root/deb/
COPY --from=gtsam_points_cuda /root/gtsam_points/build/*.deb /root/deb/
