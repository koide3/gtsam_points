FROM koide3/gtsam_docker:jammy_cuda11.8

RUN update-alternatives --install /usr/bin/ld ld /usr/bin/ld.lld 50

# resolve omp_is_initial_device-related errors
RUN sed -i '496{s/^/\/\//}' /usr/lib/llvm-14/lib/clang/14.0.0/include/omp.h

COPY . /root/gtsam_points
WORKDIR /root/gtsam_points/build
RUN rm -rf *
RUN CC=clang CXX=clang++ \
  cmake .. \
  -DBUILD_DEMO=ON \
  -DBUILD_TESTS=ON \
  -DBUILD_WITH_CUDA=ON \
  -DBUILD_EXAMPLE=ON \
  -DBUILD_WITH_CUDA_MULTIARCH=ON \
  -DCMAKE_BUILD_TYPE=Release
RUN make -j$(nproc)

CMD ["bash"]
