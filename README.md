# InBox: Recommendation with Knowledge Graph using Interest Box Embedding

This repository contains the code for the paper "InBox: Recommendation with Knowledge Graph using Interest Box Embedding," which has been accepted by VLDB.


## Running the Project
CUDA_VISIBLE_DEVICES=0 python main.py --cuda --dataset=alibaba-fashion -pre -pre_i -train -test

CUDA_VISIBLE_DEVICES=0 python main.py --cuda --dataset=yelp2018 -pre -pre_i -train -test

CUDA_VISIBLE_DEVICES=0 python main.py --cuda --dataset=last-fm -pre -pre_i -train -test

CUDA_VISIBLE_DEVICES=0 python main.py --cuda --dataset=amazon-book -pre -pre_i -train -test -pre_epoch 8

## Datasets
The project includes the following datasets:
- Alibaba Fashion
- Amazon Book
- Last.fm
- Yelp 2018

Each dataset directory contains multiple files such as `entity_list.txt`, `item_list.txt`, `kg_final.txt`, etc.
