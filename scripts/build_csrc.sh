#!/bin/bash


function check_cmd_result() {
  if [ $? -eq 0 ]; then
      echo "succeed"
  else
      echo "failed"
      exit
  fi
}

cd ./lib/csrc || exit

#cd ../ransac_voting || exit
cd ransac_voting || exit
python setup.py build_ext --inplace
check_cmd_result

cd ../nn || exit
python setup.py build_ext --inplace
check_cmd_result

cd ../fps || exit
python setup.py build_ext --inplace
check_cmd_result

cd ../uncertainty_pnp || exit
sudo apt-get install libgoogle-glog-dev
sudo apt-get install libsuitesparse-dev
sudo apt-get install libatlas-base-dev
sudo apt-get install libceres-dev
sudo apt-get install libgoogle-glog-dev
python setup.py build_ext --inplace
check_cmd_result

cd ../../ || exit
