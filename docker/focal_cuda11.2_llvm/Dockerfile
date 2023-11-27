FROM koide3/gtsam_docker:focal_cuda11.2

RUN update-alternatives --install /usr/bin/ld ld /usr/bin/ld.lld 50

COPY . /root/gtsam_ext
WORKDIR /root/gtsam_ext/build
RUN rm -rf *
RUN CC=clang CXX=clang++ \
  cmake .. \
  -DBUILD_DEMO=ON \
  -DBUILD_TESTS=ON \
  -DBUILD_EXAMPLE=ON \
  -DBUILD_WITH_CUDA=ON \
  -DBUILD_WITH_CUDA_MULTIARCH=ON \
  -DCMAKE_BUILD_TYPE=Release
RUN make -j$(nproc)

CMD ["bash"]
