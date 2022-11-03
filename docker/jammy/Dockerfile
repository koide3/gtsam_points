FROM koide3/gtsam_docker:jammy

COPY . /root/gtsam_ext
WORKDIR /root/gtsam_ext/build
RUN rm -rf *
RUN cmake .. \
  -DBUILD_DEMO=ON \
  -DBUILD_TESTS=ON \
  -DBUILD_EXAMPLE=ON \
  -DBUILD_WITH_CUDA=OFF \
  -DCMAKE_BUILD_TYPE=Release
RUN make -j$(nproc)
RUN make test

CMD ["bash"]
