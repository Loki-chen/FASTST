##  Secure Bert


mkdir -p build
cd build

cmake -DCMAKE_INSTALL_PREFIX=./install ../
enable test : 
-DCMAKE_INSTALL_PREFIX=./install ../ -FASTST_test=ON


cmake --build  --config Debug --target all --

