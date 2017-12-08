#!/bin/bash
#PBS -l nodes=1:ppn=24
#PBS -l walltime=24:00:00
#PBS -N session2_default
#PBS -A course
#PBS -q ShortQ

export THEANO_FLAGS=device=gpu2,floatX=float32

cd $PBS_O_WORKDIR

python ${HOME}/NMT_WP/WPED_with_Attention/translate.py \
	${HOME}/NMT_WP/WPED_with_Attention/param/model_wped_8m.npz  \
	${HOME}/NMT_Data/CH-EN/vocabulary/cn.8m.pkl \
	${HOME}/NMT_Data/CH-EN/vocabulary/en.8m.pkl \
	${HOME}/NMT_Data/CH-EN/source/MT03.cn.dev \
	${HOME}/NMT_WP/WPED_with_Attention/result/translate_mt03

