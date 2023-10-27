#!/bin/bash

#Please write ids of replay to download
ids=(52965522 52958192 52900827)

mkdir -p data && cd data

for id in ${ids[@]}
do
    curl -O https://www.kaggleusercontent.com/episodes/${id}.json
done
