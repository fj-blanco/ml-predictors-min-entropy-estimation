#!/bin/bash
sudo apt-get update
sudo apt-get install -y libjsoncpp-dev libmpfr-dev libdivsufsort-dev libbz2-dev libssl-dev

git clone https://github.com/usnistgov/SP800-90B_EntropyAssessment.git
cd SP800-90B_EntropyAssessment/cpp
make