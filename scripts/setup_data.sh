#!/usr/bin/env bash
# Links the three datasets into ./data. Set the three paths to where your copies live.
cd "$(dirname "$0")/.."
NSL=${NSL:-/path/to/nslkdd}; TON=${TON:-/path/to/toniot}; CIC=${CIC:-/path/to/MachineLearningCVE}
mkdir -p data/nslkdd data/toniot data/cicids
ln -sf "$NSL"/KDDTrain.txt "$NSL"/KDDTest.txt data/nslkdd/
ln -sf "$TON"/*.csv data/toniot/
ln -sf "$CIC"/*.csv data/cicids/
ls data/*
