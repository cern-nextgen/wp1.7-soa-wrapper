# wp1.7-soa-wrapper
Wrapper class with common interface for different data layouts (SoA and AoS).

## Requirements
- A machine with an nvidia GPU and the necessary drivers.
- Retrieve any further dependencies from this [script](https://github.com/cern-nextgen/wp1.7-soa-wrapper-image/blob/main/install.sh).

## Build and Run
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
Note that you still need to run the container on a machine with nvidia GPU.
Once you are within the container, you can follow the steps of the section [Build and Run](#build-and-run).
Pull the above image and run it with one of the following commands.

### Docker
```
sudo docker run -it --rm --runtime=nvidia --gpus all wp1.7-soa-wrapper:latest bash
```

### Podman
```
podman run -it --rm --security-opt=label=disable --device nvidia.com/gpu=all wp1.7-soa-wrapper:latest bash
```

## Run on kubernetes cluster of NGT
Follow the process described [here](https://ngt.docs.cern.ch/getting-started/), but in Step 6, use the following session.yaml instead.
```
apiVersion: v1
kind: Pod
metadata:
  name: session-1
  labels:
    mount-eos: "true"
    inject-oauth2-token-pipeline: "true"
  annotations:
    sidecar.istio.io/inject: "false"
spec:
  containers:
  - name: session-1
    image: registry.cern.ch/ngt-wp1.7/wp1.7-soa-wrapper:latest
    command: ["sleep", "infinity"]
    resources:
      limits:
        nvidia.com/gpu: 1
    securityContext:
      runAsUser: 0 
      runAsGroup: 0
```
In Step 9, instead of running nvidia-smi, execute the commands in the section [Build and Run](#build-and-run).


## TODO
- Default initializations
- Choose between uninitialized, default initialized, and custom initialized arrays
- Allow putting default values in skeleton struct definition, e.g. by template spcialization
- Aggregate initialization for arrays
- Don't define closures twice
- Return classical reference to struct in AoS case
- Allow classical reference when passing to a function in AoS case
