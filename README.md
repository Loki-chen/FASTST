#  Secure Bert

FaST is Fast and Secure Transformer model implement with C++, based on bert.

# Build
```bash
mkdir build & cd build

cmake -DCMAKE_INSTALL_PREFIX=./install ../
```


if you want to turn off the test sample, run:

```bash
mkdir build & cd build
-DCMAKE_INSTALL_PREFIX=./install ../ -FASTST_test=ON
```

cmake --build  --config Debug --target all --


# Acknowledgments

This repository includes code from the following external repositories:

[Microsoft/SEAL](https://github.com/microsoft/SEAL) for cryptographic tools,

[emp-toolkit/emp-tool](https://github.com/emp-toolkit/emp-tool) for Network IO,

[Microsoft/EzPC/SCI](https://github.com/Loki-chen/EzPC/tree/master/SCI) for fixed-point basic operation.