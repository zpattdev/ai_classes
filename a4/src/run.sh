#!/bin/bash

ARG_COMPILE=--${2:-'no-compile'}
ARG_BACKEND=--backend=${3:-'inductor'}

if [ "$1" = "train_gpu" ]; then
	CUDA_VISIBLE_DEVICES=0 python run.py train --train-src=./chr_en_data/train.chr --train-tgt=./chr_en_data/train.en --dev-src=./chr_en_data/dev.chr --dev-tgt=./chr_en_data/dev.en --vocab=vocab.json $ARG_COMPILE $ARG_BACKEND --gpu --lr=5e-4 --patience=1 --valid-niter=200 --batch-size=32 --dropout=.3
elif [ "$1" = "test_gpu" ]; then
        CUDA_VISIBLE_DEVICES=0 python run.py decode model.bin ./chr_en_data/test.chr ./chr_en_data/test.en ./submission/test_outputs.txt $ARG_COMPILE $ARG_BACKEND --gpu
elif [ "$1" = "train_cpu" ]; then
	python run.py train --train-src=./chr_en_data/train.chr --train-tgt=./chr_en_data/train.en --dev-src=./chr_en_data/dev.chr --dev-tgt=./chr_en_data/dev.en --vocab=vocab.json $ARG_COMPILE $ARG_BACKEND --lr=5e-4 --patience=1 --valid-niter=200 --batch-size=32 --dropout=.3
elif [ "$1" = "test_cpu" ]; then
    python run.py decode model.bin ./chr_en_data/test.chr ./chr_en_data/test.en ./submission/test_outputs.txt $ARG_COMPILE $ARG_BACKEND
elif [ "$1" = "vocab" ]; then
	python vocab.py --train-src=./chr_en_data/train.chr --train-tgt=./chr_en_data/train.en vocab.json
else
	echo "Invalid Option Selected"
fi
