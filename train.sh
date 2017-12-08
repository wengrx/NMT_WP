#!/bin/bash
#PBS -l nodes=1:ppn=20
#PBS -l walltime=168:00:00
#PBS -N session2_default
#PBS -A course
#PBS -q GpuQ

export THEANO_FLAGS=device=gpu1,floatX=float32

cd $PBS_O_WORKDIR
python ${HOME}/NMT_WP/WPED_with_Attention/train_nmt.py \
	${HOME}/NMT_Data/CH-EN/train/cn.8m.tok \
	${HOME}/NMT_Data/CH-EN/train/en.8m.tok \
	${HOME}/NMT_Data/CH-EN/vocabulary/cn.8m.pkl \
	${HOME}/NMT_Data/CH-EN/vocabulary/en.8m.pkl \
	${HOME}/NMT_Data/CH-EN/source/MT02.cn.dev \
	${HOME}/NMT_Data/CH-EN/reference/MT02/ref0 \
    ${HOME}/NMT_WP/WPED_with_Attention/param/model_hal.npz \
    ${HOME}/NMT_WP/WPED_with_Attention/param/model_wped.npz