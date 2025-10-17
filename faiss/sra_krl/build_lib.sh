mkdir build
cd build
rm * -rf
cmake ..
make -j

cd ..
mkdir -p out/lib out/include
cp build/*.so* out/lib
cp include/krl.h out/include