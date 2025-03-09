```bash
sudo apt-get update
sudo apt-get upgrade -y
sudo apt-get install cmake wget build-essential libprotobuf-dev protobuf-compiler git -y
```

```bash
git clone --depth=1 https://github.com/Tencent/ncnn.git
cd ncnn
mkdir build && cd build

cmake -D NCNN_DISABLE_RTTI=OFF -D NCNN_BUILD_TOOLS=ON -D CMAKE_BUILD_TYPE=Release ..
cmake -DCMAKE_INSTALL_PREFIX=/usr/local ..
make -j$(nproc)
sudo make install
```

```bash
sudo apt install libjpeg-dev
cd ~
wget https://raw.githubusercontent.com/nothings/stb/master/stb_image.h
```

```bash
g++ -std=c++17 -O2 -I/usr/local/include/ncnn -fopenmp yolo_ncnn.cpp -L/usr/local/lib -lncnn -o run_model
```
