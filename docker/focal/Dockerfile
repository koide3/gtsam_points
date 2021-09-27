FROM ubuntu:focal

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
  && apt-get install --no-install-recommends -y \
  && apt-get install --no-install-recommends -y \
      wget nano build-essential \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

RUN apt-get update \
  && apt-get install --no-install-recommends -y \
      git cmake libeigen3-dev ca-certificates

RUN git clone https://github.com/borglab/gtsam.git /root/gtsam

RUN apt-get update \
  && apt-get install --no-install-recommends -y \
      libboost-all-dev

WORKDIR /root/gtsam/build
RUN cmake .. \
  -DGTSAM_USE_SYSTEM_EIGEN=ON \
  -DGTSAM_WITH_TBB=OFF \
  -DGTSAM_BUILD_WITH_MARCH_NATIVE=OFF \
  -DGTSAM_BUILD_TESTS=OFF \
  -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF
RUN make -j$(nproc) && make install


COPY . /root/gtsam_ext
WORKDIR /root/gtsam_ext/build
RUN cmake ..

# RUN mkdir -p /root/catkin_ws/src
# WORKDIR /root/catkin_ws/src
# RUN /bin/bash -c '. /opt/ros/melodic/setup.bash; catkin_init_workspace'
# 
# COPY . /root/catkin_ws/src/fast_gicp/
# WORKDIR /root/catkin_ws/src/fast_gicp
# RUN git submodule init && git submodule update
# 
# WORKDIR /root/catkin_ws
# RUN /bin/bash -c '. /opt/ros/melodic/setup.bash; catkin_make'
# RUN sed -i "6i source \"/root/catkin_ws/devel/setup.bash\"" /ros_entrypoint.sh

WORKDIR /

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
