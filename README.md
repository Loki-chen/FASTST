#  Secure Bert

FASTLMPI is Fast And Secure Transformer model implement with C++, based on bert.

# Build
```bash
mkdir build & cd build
cmake -DCMAKE_INSTALL_PREFIX=./install ../
```


if you want to turn off the test sample, run:

```bash
mkdir build & cd build
cmake -DCMAKE_INSTALL_PREFIX=./install ../ -FASTST_test=ON
cmake --build  --config Debug --target all --
```

or just run:

```bash
./setup_and_build.sh
```


# Acknowledgments

This repository includes code from the following external repositories:

[Microsoft/SEAL](https://github.com/microsoft/SEAL) for cryptographic tools,

[emp-toolkit/emp-tool](https://github.com/emp-toolkit/emp-tool) for Network IO,

[Microsoft/EzPC/SCI](https://github.com/Loki-chen/EzPC/tree/master/SCI) for fixed-point basic operation.


# Other

The end-to-end performance is the aggregation of themicrobenchmarks, end-to-end code is coming soon....