#!/bin/bash

cd $PBS_O_WORKDIR

python ${HOME}/NMT_WP/WPED_with_Attention/translate_allfile.py \
    --gpu gpu2 \
    --floatX float32 \
    --process-limit 4 \
    --multi-bleu-perl ${HOME}/Utils/multi-bleu.perl \
    --script ${HOME}/NMT_WP/WPED_with_Attention/translate.py \
    --model ${HOME}/NMT_WP/WPED_with_Attention/param/model_wped_8m.npz \
    ${HOME}/NMT_Data/CH-EN/vocabulary/cn.8m.pkl \
    ${HOME}/NMT_Data/CH-EN/vocabulary/en.8m.pkl \
    ${HOME}/NMT_Data/CH-EN/source/MT02.cn.dev,${HOME}/NMT_Data/CH-EN/source/MT03.cn.dev,${HOME}/NMT_Data/CH-EN/source/MT04.cn.dev,${HOME}/NMT_Data/CH-EN/source/MT05.cn.dev \
    ${HOME}/NMT_Data/CH-EN/reference/MT02/ref,${HOME}/NMT_Data/CH-EN/reference/MT03/ref,${HOME}/NMT_Data/CH-EN/reference/MT04/ref,${HOME}/NMT_Data/CH-EN/reference/MT05/ref \
    ${HOME}/NMT_WP/WPED_with_Attention/result/transmt02_wped,${HOME}/NMT_WP/WPED_with_Attention/result/transmt03_wped,${HOME}/NMT_WP/WPED_with_Attention/result/transmt04_wped,${HOME}/NMT_WP/WPED_with_Attention/result/transmt05_wped \
    ${HOME}/NMT_WP/WPED_with_Attention/result/save_bleu