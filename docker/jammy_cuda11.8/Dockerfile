FROM koide3/gtsam_docker:jammy_cuda11.8

COPY . /root/gtsam_points
WORKDIR /root/gtsam_points/build
RUN rm -rf *
RUN cmake .. \
  -DBUILD_DEMO=ON \
  -DBUILD_TESTS=ON \
  -DBUILD_WITH_CUDA=ON \
  -DBUILD_EXAMPLE=ON \
  -DBUILD_WITH_CUDA_MULTIARCH=ON \
  -DCMAKE_BUILD_TYPE=Release
RUN make -j$(nproc)

CMD ["bash"]
