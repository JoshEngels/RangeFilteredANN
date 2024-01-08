#!/bin/bash
cd ~/ParlayANN/algorithms/HCNNG
make clean
make

P=/ssd1/anndata
G=/ssd1/results
# BP=$P/text2image1B
# BG=$G/text2image1B
BP=$P/bigann
BG=$G/bigann
echo $BP
echo $BG
for i in 192; do
    # PARLAY_NUM_THREADS=$i nohup ./neighbors -R 64 -L 128 -file_type bin -data_type uint8 -dist_func Euclidian -base_path $BP/base.1B.u8bin.crop_nb_1000000 >> test_parallel.out
    # PARLAY_NUM_THREADS=$i nohup ./neighbors -R 64 -L 128 -file_type bin -data_type uint8 -dist_func Euclidian -base_path $BP/base.1B.u8bin.crop_nb_1000000 >> test_parallel.out
    # ./neighbors -R 64 -L 128 -file_type bin -data_type uint8 -dist_func Euclidian -base_path /ssd1/anndata/bigann/base.1B.u8bin.crop_nb_1000000
    PARLAY_NUM_THREADS=$i nohup ./neighbors -a 1000 -R 50 -L 20 -k 200 -Q 250 -b 1 -file_type bin -data_type uint8 -dist_func mips -base_path $BP/data/yfcc100M/base.10M.u8bin.crop_nb_10000000 >> test_parallel.out
done
