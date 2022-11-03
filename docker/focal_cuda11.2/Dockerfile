FROM koide3/gtsam_docker:focal_cuda11.2

COPY . /root/gtsam_ext
WORKDIR /root/gtsam_ext/build
RUN rm -rf *
RUN cmake .. \
  -DBUILD_DEMO=ON \
  -DBUILD_TESTS=ON \
  -DBUILD_WITH_CUDA=ON \
  -DBUILD_EXAMPLE=ON \
  -DCMAKE_BUILD_TYPE=Release
RUN make -j$(nproc)

CMD ["bash"]
