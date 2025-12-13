#!/bin/bash

ARG_COMPILE=--${2:-'no-compile'}
ARG_BACKEND=--backend=${3:-'inductor'}

if [ "$1" = "vanilla_finetune_without_pretrain" ]; then
	python run.py $ARG_COMPILE $ARG_BACKEND --function=finetune --variant=vanilla --pretrain_corpus_path=./data/wiki.txt --writing_params_path=./submission/vanilla.model.params --finetune_corpus_path=./data/birth_places_train.tsv
elif [ "$1" = "vanilla_eval_dev_without_pretrain" ]; then
	if [ -f ./submission/vanilla.model.params ]; then
    	python run.py $ARG_COMPILE $ARG_BACKEND --function=evaluate --variant=vanilla --pretrain_corpus_path=./data/wiki.txt --reading_params_path=./submission/vanilla.model.params --eval_corpus_path=./data/birth_dev.tsv --outputs_path=./submission/vanilla.nopretrain.dev.predictions
	else
		echo "'./submission/vanilla.model.params' does not exist. Please run './run.sh vanilla_finetune_without_pretrain' on the VM to create this file."
	fi
elif [ "$1" = "vanilla_eval_test_without_pretrain" ]; then
	if [ -f ./submission/vanilla.model.params ]; then
		python run.py $ARG_COMPILE $ARG_BACKEND --function=evaluate --variant=vanilla --pretrain_corpus_path=./data/wiki.txt --reading_params_path=./submission/vanilla.model.params --eval_corpus_path=./data/birth_test_inputs.tsv --outputs_path=./submission/vanilla.nopretrain.test.predictions
	else
		echo "'./submission/vanilla.model.params' does not exist. Please run './run.sh vanilla_finetune_without_pretrain' on the VM to create this file."
	fi
elif [ "$1" = "vanilla_pretrain" ]; then
	echo "Starting Vanilla Pretrain: ~ 1 Hours"
    python run.py $ARG_COMPILE $ARG_BACKEND --function=pretrain --variant=vanilla --pretrain_corpus_path=./data/wiki.txt --writing_params_path=./submission/vanilla.pretrain.params
elif [ "$1" = "vanilla_finetune_with_pretrain" ]; then
	if [ -f ./submission/vanilla.pretrain.params ]; then
		python run.py $ARG_COMPILE $ARG_BACKEND --function=finetune --variant=vanilla --pretrain_corpus_path=./data/wiki.txt --reading_params_path=./submission/vanilla.pretrain.params --writing_params_path=./submission/vanilla.finetune.params --finetune_corpus_path=./data/birth_places_train.tsv
	else
		echo "'./submission/vanilla.pretrain.params' does not exist. Please run './run.sh vanilla_pretrain' on the VM to create this file. Note: will take around 1 hours."
	fi
elif [ "$1" = "vanilla_eval_dev_with_pretrain" ]; then
	if [ -f ./submission/vanilla.finetune.params ]; then
		python run.py $ARG_COMPILE $ARG_BACKEND --function=evaluate --variant=vanilla --pretrain_corpus_path=./data/wiki.txt --reading_params_path=./submission/vanilla.finetune.params --eval_corpus_path=./data/birth_dev.tsv --outputs_path=./submission/vanilla.pretrain.dev.predictions
	else
		echo "'./submission/vanilla.finetune.params' does not exist. Please run './run.sh vanilla_finetune_with_pretrain' on the VM to create this file."
	fi
elif [ "$1" = "vanilla_eval_test_with_pretrain" ]; then
	if [ -f ./submission/vanilla.finetune.params ]; then
		python run.py $ARG_COMPILE $ARG_BACKEND --function=evaluate --variant=vanilla --pretrain_corpus_path=./data/wiki.txt --reading_params_path=./submission/vanilla.finetune.params --eval_corpus_path=./data/birth_test_inputs.tsv --outputs_path=./submission/vanilla.pretrain.test.predictions
	else
		echo "'./submission/vanilla.finetune.params' does not exist. Please run './run.sh vanilla_finetune_with_pretrain' on the VM to create this file."
	fi
elif [ "$1" = "rope_pretrain" ]; then
	echo "Starting Rope Pretrain: ~ 1 Hours"
	python run.py $ARG_COMPILE $ARG_BACKEND --function=pretrain --variant=rope --pretrain_corpus_path=./data/wiki.txt --writing_params_path=./submission/rope.pretrain.params	
elif [ "$1" = "rope_finetune_with_pretrain" ]; then
	if [ -f ./submission/rope.pretrain.params ]; then
		python run.py $ARG_COMPILE $ARG_BACKEND --function=finetune --variant=rope --pretrain_corpus_path=./data/wiki.txt --reading_params_path=./submission/rope.pretrain.params --writing_params_path=./submission/rope.finetune.params --finetune_corpus_path=./data/birth_places_train.tsv
	else
		echo "'./submission/rope.pretrain.params' does not exist. Please run './run.sh rope_finetune_with_pretrain' on the VM to create this file. Note: will take around 1 hours."
	fi
elif [ "$1" = "rope_eval_dev_with_pretrain" ]; then
	if [ -f ./submission/rope.finetune.params ]; then
		python run.py $ARG_COMPILE $ARG_BACKEND --function=evaluate --variant=rope --pretrain_corpus_path=./data/wiki.txt --reading_params_path=./submission/rope.finetune.params --eval_corpus_path=./data/birth_dev.tsv --outputs_path=./submission/rope.pretrain.dev.predictions	
	else
		echo "'./submission/rope.finetune.params' does not exist. Please run './run.sh vanilla_finetune_with_pretrain' on the VM to create this file."
	fi
elif [ "$1" = "rope_eval_test_with_pretrain" ]; then
	if [ -f ./submission/rope.finetune.params ]; then
		python run.py $ARG_COMPILE $ARG_BACKEND --function=evaluate --variant=rope --pretrain_corpus_path=./data/wiki.txt --reading_params_path=./submission/rope.finetune.params --eval_corpus_path=./data/birth_test_inputs.tsv --outputs_path=./submission/rope.pretrain.test.predictions	
	else
		echo "'./submission/rope.finetune.params' does not exist. Please run './run.sh vanilla_finetune_with_pretrain' on the VM to create this file."
	fi 
else
	echo "Invalid Option Selected. Only Options Available Are:"
	echo "=============================================================="
	echo "./run.sh vanilla_finetune_without_pretrain"
	echo "./run.sh vanilla_eval_dev_without_pretrain"
	echo "./run.sh vanilla_eval_test_without_pretrain"
	echo "------------------------------------------------------------"
	echo "./run.sh vanilla_pretrain"
	echo "./run.sh vanilla_finetune_with_pretrain"
	echo "./run.sh vanilla_eval_dev_with_pretrain"
	echo "./run.sh vanilla_eval_test_with_pretrain"
	echo "------------------------------------------------------------"
	echo "./run.sh rope_pretrain"
	echo "./run.sh rope_finetune_with_pretrain"
	echo "./run.sh rope_eval_dev_with_pretrain"
	echo "./run.sh rope_eval_test_with_pretrain"
	echo "------------------------------------------------------------"
fi