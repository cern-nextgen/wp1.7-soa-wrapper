# wp1.7-soa-wrapper
Wrapper class to interface with different data layouts (SoA and AoS).

## Build and Run with CMake (Linux)
Open a Linux terminal, navigate to the cloned directory of this repository, and run the following commands.
```
mkdir build
cd build
cmake ..
make
./wrapper

## TODO
- Default initializations
- Choose between uninitialized, default initialized, and custom initialized arrays
- Allow putting default values in skeleton struct definition, e.g. by template spcialization
- Agregate initialization for arrays
- Don't define closures twice
- Return classical reference to struct in AoS case
- Allow classical reference wehn passing to a function in AoS case
```
