#!/bin/bash
sudo apt-get update
echo "Installing system dependencies"
sudo apt-get install -y cmake libgtest-dev
sudo apt-get install -y autoconf libtool pkg-config libgdal-dev libxerces-c-dev
sudo apt-get install -y libproj-dev libfox-1.6-dev libxml2-dev libxslt1-dev
sudo apt-get install -y build-essential curl unzip flex bison

echo "Installing sumo binaries"
mkdir -p $HOME/sumo_binaries/bin
pushd $HOME/sumo_binaries/bin
wget https://akreidieh.s3.amazonaws.com/sumo/flow-0.4.0/binaries-ubuntu1804.tar.xz
tar -xf binaries-ubuntu1804.tar.xz
rm binaries-ubuntu1804.tar.xz
chmod +x *
popd
echo 'export PATH="$HOME/sumo_binaries/bin:$PATH"' >> ~/.bashrc
echo 'export SUMO_HOME="$HOME/sumo_binaries/bin"' >> ~/.bashrc
