#!/bin/bash
sudo apt-get update
echo "Installing system dependencies"
sudo apt-get install -y cmake libopenmpi-dev zlib1g-dev
sudo apt-get install -y autoconf build-essential libtool
sudo apt-get install -y libxerces-c3.1 libxerces-c3-dev libproj-dev
sudo apt-get install -y proj-bin proj-data libgdal1-dev
sudo apt-get install -y libfox-1.6-0 libfox-1.6-dev

echo "Installing sumo binaries"
mkdir -p $HOME/sumo_binaries/bin
pushd $HOME/sumo_binaries/bin
wget https://akreidieh.s3.amazonaws.com/sumo/flow-0.4.0/binaries-ubuntu1404.tar.xz
tar -xf binaries-ubuntu1404.tar.xz
rm binaries-ubuntu1404.tar.xz
chmod +x *
popd
echo 'export PATH="$HOME/sumo_binaries/bin:$PATH"' >> ~/.bashrc
echo 'export SUMO_HOME="$HOME/sumo_binaries/bin"' >> ~/.bashrc
