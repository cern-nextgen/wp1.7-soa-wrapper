# wp1.7-soa-wrapper
Single header library for a common interface for different data layouts (SoA and AoS).

## Install the Library
You need to have git and CMake installed.
```
git clone https://github.com/cern-nextgen/wp1.7-soa-wrapper.git
cd wp1.7-soa-wrapper
cmake -B build
cmake --build build --target install
```

## Run the Unit Tests
This needs additinally a C++20 compiler. It will skip the GPU tests unless you have a Nvidia GPU and the CUDA toolkit installed.
```
git clone https://github.com/cern-nextgen/wp1.7-soa-wrapper.git
cd wp1.7-soa-wrapper
cmake -B build -DBUILD_TESTING=ON
cmake --build build
ctest --test-dir build/tests
```

## Container
The following container contains the [nvidia container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/sample-workload.html) and the dependencies to build and run the unit tests (and the documentation).
```
registry.cern.ch/ngt-wp1.7/wp1.7-soa-wrapper:latest
```
The corresponding Dockerfile can be found here [wp1.7-soa-wrapper-image](https://github.com/cern-nextgen/wp1.7-soa-wrapper-image).
Note that you still need to run the container on a machine with nvidia GPU.
Once you are within the container, you can follow the steps of the section [Build and Run](#build-and-run).
Pull the above image and run it with one of the following commands.

### Docker
```
sudo docker pull registry.cern.ch/ngt-wp1.7/wp1.7-soa-wrapper:latest
sudo docker run -it --rm --runtime=nvidia --gpus all wp1.7-soa-wrapper:latest bash
```

### Podman
```
podman pull registry.cern.ch/ngt-wp1.7/wp1.7-soa-wrapper:latest
podman run -it --rm --security-opt=label=disable --device nvidia.com/gpu=all wp1.7-soa-wrapper:latest bash
```

## Kubernetes Cluster of NGT
Follow the process described [here](https://ngt.docs.cern.ch/getting-started/), but in Step 6, use the following session.yaml instead.
```
apiVersion: v1
kind: Pod
metadata:
  name: soa-wrapper
spec:
  containers:
  - name: soa-wrapper
    image: registry.cern.ch/ngt-wp1.7/wp1.7-soa-wrapper:latest
    command: ["sleep", "infinity"]
    resources:
      limits:
        nvidia.com/gpu: 1
    securityContext:
      runAsUser: 0
      runAsGroup: 0
  nodeSelector:
    nvidia.com/gpu.product: NVIDIA-H100-NVL
```

## TODO
- Default initializations
- Choose between uninitialized, default initialized, and custom initialized arrays
- Allow putting default values in skeleton struct definition, e.g. by template specialization
- Aggregate initialization for arrays
- Don't define closures twice
- Return classical reference to struct in AoS case
- Allow classical reference when passing to a function in AoS case
