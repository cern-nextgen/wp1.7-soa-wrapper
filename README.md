# wp1.7-soa-wrapper
Wrapper class with common interface for different data layouts (SoA and AoS).

## Requirements
- A machine with an Nvidia GPU and the necessary drivers

## Build and Run the tests (Linux)
```
git clone https://github.com/cern-nextgen/wp1.7-soa-wrapper.git
cd wp1.7-soa-wrapper
cmake -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

## Container
The following container contains the [nvidia container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/sample-workload.html) and the dependencies to build the unit tests.
```
registry.cern.ch/ngt-wp1.7/wp1.7-soa-wrapper:latest
```
The corresponding Dockerfile can be found here [wp1.7-soa-wrapper-image](https://github.com/cern-nextgen/wp1.7-soa-wrapper-image).
Note that you still need to run the container on a mchine with Nvidia GPU.
Once you are within the container, you can follow the steps of the section Build and Run the tests (Linux).
Pull the above image and run it with one of the following commands.

### Docker
```
sudo docker run -it --rm --runtime=nvidia --gpus all wp1.7-soa-wrapper:latest bash
```

### Podman
```
podman run -it --rm --security-opt=label=disable --device nvidia.com/gpu=all wp1.7-soa-wrapper:latest bash
```

## TODO
- Default initializations
- Choose between uninitialized, default initialized, and custom initialized arrays
- Allow putting default values in skeleton struct definition, e.g. by template spcialization
- Agregate initialization for arrays
- Don't define closures twice
- Return classical reference to struct in AoS case
- Allow classical reference when passing to a function in AoS case
