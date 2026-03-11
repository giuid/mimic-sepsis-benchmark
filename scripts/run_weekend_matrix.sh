#!/bin/bash
# Sepsis Matrix Completion Script - Weekend Run

echo 'Launching Run 1/92: CUDA_VISIBLE_DEVICES=4 python scripts/train_sepsis_benchmarks.py --task ihm --model transformer --feature_subset full --epochs 30 --batch_size 64'
CUDA_VISIBLE_DEVICES=4 python scripts/train_sepsis_benchmarks.py --task ihm --model transformer --feature_subset full --epochs 30 --batch_size 64
sleep 10

echo 'Launching Run 2/92: CUDA_VISIBLE_DEVICES=5 python scripts/train_sepsis_benchmarks.py --task ihm --model transformer --feature_subset full --epochs 30 --batch_size 64 --use_kgi --kgi_mode dki'
CUDA_VISIBLE_DEVICES=5 python scripts/train_sepsis_benchmarks.py --task ihm --model transformer --feature_subset full --epochs 30 --batch_size 64 --use_kgi --kgi_mode dki
sleep 10

echo 'Launching Run 3/92: CUDA_VISIBLE_DEVICES=4 python scripts/train_sepsis_benchmarks.py --task ihm --model transformer --feature_subset full --epochs 30 --batch_size 64 --use_kgi --kgi_mode dgi'
CUDA_VISIBLE_DEVICES=4 python scripts/train_sepsis_benchmarks.py --task ihm --model transformer --feature_subset full --epochs 30 --batch_size 64 --use_kgi --kgi_mode dgi
sleep 10

echo 'Launching Run 4/92: CUDA_VISIBLE_DEVICES=5 python scripts/train_sepsis_benchmarks.py --task ihm --model transformer --feature_subset no_treatments --epochs 30 --batch_size 64'
CUDA_VISIBLE_DEVICES=5 python scripts/train_sepsis_benchmarks.py --task ihm --model transformer --feature_subset no_treatments --epochs 30 --batch_size 64
sleep 10

echo 'Launching Run 5/92: CUDA_VISIBLE_DEVICES=4 python scripts/train_sepsis_benchmarks.py --task ihm --model transformer --feature_subset no_treatments --epochs 30 --batch_size 64 --use_kgi --kgi_mode dki'
CUDA_VISIBLE_DEVICES=4 python scripts/train_sepsis_benchmarks.py --task ihm --model transformer --feature_subset no_treatments --epochs 30 --batch_size 64 --use_kgi --kgi_mode dki
sleep 10

echo 'Launching Run 6/92: CUDA_VISIBLE_DEVICES=5 python scripts/train_sepsis_benchmarks.py --task ihm --model transformer --feature_subset no_treatments --epochs 30 --batch_size 64 --use_kgi --kgi_mode dgi'
CUDA_VISIBLE_DEVICES=5 python scripts/train_sepsis_benchmarks.py --task ihm --model transformer --feature_subset no_treatments --epochs 30 --batch_size 64 --use_kgi --kgi_mode dgi
sleep 10

echo 'Launching Run 7/92: CUDA_VISIBLE_DEVICES=4 python scripts/train_sepsis_benchmarks.py --task ihm --model transformer --feature_subset core --epochs 30 --batch_size 64'
CUDA_VISIBLE_DEVICES=4 python scripts/train_sepsis_benchmarks.py --task ihm --model transformer --feature_subset core --epochs 30 --batch_size 64
sleep 10

echo 'Launching Run 8/92: CUDA_VISIBLE_DEVICES=5 python scripts/train_sepsis_benchmarks.py --task ihm --model transformer --feature_subset core --epochs 30 --batch_size 64 --use_kgi --kgi_mode dki'
CUDA_VISIBLE_DEVICES=5 python scripts/train_sepsis_benchmarks.py --task ihm --model transformer --feature_subset core --epochs 30 --batch_size 64 --use_kgi --kgi_mode dki
sleep 10

echo 'Launching Run 9/92: CUDA_VISIBLE_DEVICES=4 python scripts/train_sepsis_benchmarks.py --task ihm --model transformer --feature_subset core --epochs 30 --batch_size 64 --use_kgi --kgi_mode dgi'
CUDA_VISIBLE_DEVICES=4 python scripts/train_sepsis_benchmarks.py --task ihm --model transformer --feature_subset core --epochs 30 --batch_size 64 --use_kgi --kgi_mode dgi
sleep 10

echo 'Launching Run 10/92: CUDA_VISIBLE_DEVICES=5 python scripts/train_sepsis_benchmarks.py --task ihm --model transformer --feature_subset emergency --epochs 30 --batch_size 64'
CUDA_VISIBLE_DEVICES=5 python scripts/train_sepsis_benchmarks.py --task ihm --model transformer --feature_subset emergency --epochs 30 --batch_size 64
sleep 10

echo 'Launching Run 11/92: CUDA_VISIBLE_DEVICES=4 python scripts/train_sepsis_benchmarks.py --task ihm --model transformer --feature_subset emergency --epochs 30 --batch_size 64 --use_kgi --kgi_mode dki'
CUDA_VISIBLE_DEVICES=4 python scripts/train_sepsis_benchmarks.py --task ihm --model transformer --feature_subset emergency --epochs 30 --batch_size 64 --use_kgi --kgi_mode dki
sleep 10

echo 'Launching Run 12/92: CUDA_VISIBLE_DEVICES=5 python scripts/train_sepsis_benchmarks.py --task ihm --model transformer --feature_subset emergency --epochs 30 --batch_size 64 --use_kgi --kgi_mode dgi'
CUDA_VISIBLE_DEVICES=5 python scripts/train_sepsis_benchmarks.py --task ihm --model transformer --feature_subset emergency --epochs 30 --batch_size 64 --use_kgi --kgi_mode dgi
sleep 10

echo 'Launching Run 13/92: CUDA_VISIBLE_DEVICES=4 python scripts/train_sepsis_benchmarks.py --task los --model transformer --feature_subset full --epochs 30 --batch_size 64'
CUDA_VISIBLE_DEVICES=4 python scripts/train_sepsis_benchmarks.py --task los --model transformer --feature_subset full --epochs 30 --batch_size 64
sleep 10

echo 'Launching Run 14/92: CUDA_VISIBLE_DEVICES=5 python scripts/train_sepsis_benchmarks.py --task los --model transformer --feature_subset full --epochs 30 --batch_size 64 --use_kgi --kgi_mode dki'
CUDA_VISIBLE_DEVICES=5 python scripts/train_sepsis_benchmarks.py --task los --model transformer --feature_subset full --epochs 30 --batch_size 64 --use_kgi --kgi_mode dki
sleep 10

echo 'Launching Run 15/92: CUDA_VISIBLE_DEVICES=4 python scripts/train_sepsis_benchmarks.py --task los --model transformer --feature_subset full --epochs 30 --batch_size 64 --use_kgi --kgi_mode dgi'
CUDA_VISIBLE_DEVICES=4 python scripts/train_sepsis_benchmarks.py --task los --model transformer --feature_subset full --epochs 30 --batch_size 64 --use_kgi --kgi_mode dgi
sleep 10

echo 'Launching Run 16/92: CUDA_VISIBLE_DEVICES=5 python scripts/train_sepsis_benchmarks.py --task los --model transformer --feature_subset no_treatments --epochs 30 --batch_size 64'
CUDA_VISIBLE_DEVICES=5 python scripts/train_sepsis_benchmarks.py --task los --model transformer --feature_subset no_treatments --epochs 30 --batch_size 64
sleep 10

echo 'Launching Run 17/92: CUDA_VISIBLE_DEVICES=4 python scripts/train_sepsis_benchmarks.py --task los --model transformer --feature_subset no_treatments --epochs 30 --batch_size 64 --use_kgi --kgi_mode dki'
CUDA_VISIBLE_DEVICES=4 python scripts/train_sepsis_benchmarks.py --task los --model transformer --feature_subset no_treatments --epochs 30 --batch_size 64 --use_kgi --kgi_mode dki
sleep 10

echo 'Launching Run 18/92: CUDA_VISIBLE_DEVICES=5 python scripts/train_sepsis_benchmarks.py --task los --model transformer --feature_subset no_treatments --epochs 30 --batch_size 64 --use_kgi --kgi_mode dgi'
CUDA_VISIBLE_DEVICES=5 python scripts/train_sepsis_benchmarks.py --task los --model transformer --feature_subset no_treatments --epochs 30 --batch_size 64 --use_kgi --kgi_mode dgi
sleep 10

echo 'Launching Run 19/92: CUDA_VISIBLE_DEVICES=4 python scripts/train_sepsis_benchmarks.py --task los --model transformer --feature_subset core --epochs 30 --batch_size 64'
CUDA_VISIBLE_DEVICES=4 python scripts/train_sepsis_benchmarks.py --task los --model transformer --feature_subset core --epochs 30 --batch_size 64
sleep 10

echo 'Launching Run 20/92: CUDA_VISIBLE_DEVICES=5 python scripts/train_sepsis_benchmarks.py --task los --model transformer --feature_subset core --epochs 30 --batch_size 64 --use_kgi --kgi_mode dki'
CUDA_VISIBLE_DEVICES=5 python scripts/train_sepsis_benchmarks.py --task los --model transformer --feature_subset core --epochs 30 --batch_size 64 --use_kgi --kgi_mode dki
sleep 10

echo 'Launching Run 21/92: CUDA_VISIBLE_DEVICES=4 python scripts/train_sepsis_benchmarks.py --task los --model transformer --feature_subset core --epochs 30 --batch_size 64 --use_kgi --kgi_mode dgi'
CUDA_VISIBLE_DEVICES=4 python scripts/train_sepsis_benchmarks.py --task los --model transformer --feature_subset core --epochs 30 --batch_size 64 --use_kgi --kgi_mode dgi
sleep 10

echo 'Launching Run 22/92: CUDA_VISIBLE_DEVICES=5 python scripts/train_sepsis_benchmarks.py --task los --model transformer --feature_subset emergency --epochs 30 --batch_size 64'
CUDA_VISIBLE_DEVICES=5 python scripts/train_sepsis_benchmarks.py --task los --model transformer --feature_subset emergency --epochs 30 --batch_size 64
sleep 10

echo 'Launching Run 23/92: CUDA_VISIBLE_DEVICES=4 python scripts/train_sepsis_benchmarks.py --task los --model transformer --feature_subset emergency --epochs 30 --batch_size 64 --use_kgi --kgi_mode dki'
CUDA_VISIBLE_DEVICES=4 python scripts/train_sepsis_benchmarks.py --task los --model transformer --feature_subset emergency --epochs 30 --batch_size 64 --use_kgi --kgi_mode dki
sleep 10

echo 'Launching Run 24/92: CUDA_VISIBLE_DEVICES=5 python scripts/train_sepsis_benchmarks.py --task los --model transformer --feature_subset emergency --epochs 30 --batch_size 64 --use_kgi --kgi_mode dgi'
CUDA_VISIBLE_DEVICES=5 python scripts/train_sepsis_benchmarks.py --task los --model transformer --feature_subset emergency --epochs 30 --batch_size 64 --use_kgi --kgi_mode dgi
sleep 10

echo 'Launching Run 25/92: CUDA_VISIBLE_DEVICES=4 python scripts/train_sepsis_benchmarks.py --task ss --model transformer --feature_subset full --epochs 30 --batch_size 64'
CUDA_VISIBLE_DEVICES=4 python scripts/train_sepsis_benchmarks.py --task ss --model transformer --feature_subset full --epochs 30 --batch_size 64
sleep 10

echo 'Launching Run 26/92: CUDA_VISIBLE_DEVICES=5 python scripts/train_sepsis_benchmarks.py --task ss --model transformer --feature_subset full --epochs 30 --batch_size 64 --use_kgi --kgi_mode dki'
CUDA_VISIBLE_DEVICES=5 python scripts/train_sepsis_benchmarks.py --task ss --model transformer --feature_subset full --epochs 30 --batch_size 64 --use_kgi --kgi_mode dki
sleep 10

echo 'Launching Run 27/92: CUDA_VISIBLE_DEVICES=4 python scripts/train_sepsis_benchmarks.py --task ss --model transformer --feature_subset full --epochs 30 --batch_size 64 --use_kgi --kgi_mode dgi'
CUDA_VISIBLE_DEVICES=4 python scripts/train_sepsis_benchmarks.py --task ss --model transformer --feature_subset full --epochs 30 --batch_size 64 --use_kgi --kgi_mode dgi
sleep 10

echo 'Launching Run 28/92: CUDA_VISIBLE_DEVICES=5 python scripts/train_sepsis_benchmarks.py --task ss --model transformer --feature_subset no_treatments --epochs 30 --batch_size 64'
CUDA_VISIBLE_DEVICES=5 python scripts/train_sepsis_benchmarks.py --task ss --model transformer --feature_subset no_treatments --epochs 30 --batch_size 64
sleep 10

echo 'Launching Run 29/92: CUDA_VISIBLE_DEVICES=4 python scripts/train_sepsis_benchmarks.py --task ss --model transformer --feature_subset no_treatments --epochs 30 --batch_size 64 --use_kgi --kgi_mode dki'
CUDA_VISIBLE_DEVICES=4 python scripts/train_sepsis_benchmarks.py --task ss --model transformer --feature_subset no_treatments --epochs 30 --batch_size 64 --use_kgi --kgi_mode dki
sleep 10

echo 'Launching Run 30/92: CUDA_VISIBLE_DEVICES=5 python scripts/train_sepsis_benchmarks.py --task ss --model transformer --feature_subset no_treatments --epochs 30 --batch_size 64 --use_kgi --kgi_mode dgi'
CUDA_VISIBLE_DEVICES=5 python scripts/train_sepsis_benchmarks.py --task ss --model transformer --feature_subset no_treatments --epochs 30 --batch_size 64 --use_kgi --kgi_mode dgi
sleep 10

echo 'Launching Run 31/92: CUDA_VISIBLE_DEVICES=4 python scripts/train_sepsis_benchmarks.py --task ss --model transformer --feature_subset core --epochs 30 --batch_size 64'
CUDA_VISIBLE_DEVICES=4 python scripts/train_sepsis_benchmarks.py --task ss --model transformer --feature_subset core --epochs 30 --batch_size 64
sleep 10

echo 'Launching Run 32/92: CUDA_VISIBLE_DEVICES=5 python scripts/train_sepsis_benchmarks.py --task ss --model transformer --feature_subset core --epochs 30 --batch_size 64 --use_kgi --kgi_mode dki'
CUDA_VISIBLE_DEVICES=5 python scripts/train_sepsis_benchmarks.py --task ss --model transformer --feature_subset core --epochs 30 --batch_size 64 --use_kgi --kgi_mode dki
sleep 10

echo 'Launching Run 33/92: CUDA_VISIBLE_DEVICES=4 python scripts/train_sepsis_benchmarks.py --task ss --model transformer --feature_subset core --epochs 30 --batch_size 64 --use_kgi --kgi_mode dgi'
CUDA_VISIBLE_DEVICES=4 python scripts/train_sepsis_benchmarks.py --task ss --model transformer --feature_subset core --epochs 30 --batch_size 64 --use_kgi --kgi_mode dgi
sleep 10

echo 'Launching Run 34/92: CUDA_VISIBLE_DEVICES=5 python scripts/train_sepsis_benchmarks.py --task ss --model transformer --feature_subset emergency --epochs 30 --batch_size 64'
CUDA_VISIBLE_DEVICES=5 python scripts/train_sepsis_benchmarks.py --task ss --model transformer --feature_subset emergency --epochs 30 --batch_size 64
sleep 10

echo 'Launching Run 35/92: CUDA_VISIBLE_DEVICES=4 python scripts/train_sepsis_benchmarks.py --task ss --model transformer --feature_subset emergency --epochs 30 --batch_size 64 --use_kgi --kgi_mode dki'
CUDA_VISIBLE_DEVICES=4 python scripts/train_sepsis_benchmarks.py --task ss --model transformer --feature_subset emergency --epochs 30 --batch_size 64 --use_kgi --kgi_mode dki
sleep 10

echo 'Launching Run 36/92: CUDA_VISIBLE_DEVICES=5 python scripts/train_sepsis_benchmarks.py --task ss --model transformer --feature_subset emergency --epochs 30 --batch_size 64 --use_kgi --kgi_mode dgi'
CUDA_VISIBLE_DEVICES=5 python scripts/train_sepsis_benchmarks.py --task ss --model transformer --feature_subset emergency --epochs 30 --batch_size 64 --use_kgi --kgi_mode dgi
sleep 10

echo 'Launching Run 37/92: CUDA_VISIBLE_DEVICES=4 python scripts/train_sepsis_benchmarks.py --task vr --model transformer --feature_subset full --epochs 30 --batch_size 64'
CUDA_VISIBLE_DEVICES=4 python scripts/train_sepsis_benchmarks.py --task vr --model transformer --feature_subset full --epochs 30 --batch_size 64
sleep 10

echo 'Launching Run 38/92: CUDA_VISIBLE_DEVICES=5 python scripts/train_sepsis_benchmarks.py --task vr --model transformer --feature_subset full --epochs 30 --batch_size 64 --use_kgi --kgi_mode dki'
CUDA_VISIBLE_DEVICES=5 python scripts/train_sepsis_benchmarks.py --task vr --model transformer --feature_subset full --epochs 30 --batch_size 64 --use_kgi --kgi_mode dki
sleep 10

echo 'Launching Run 39/92: CUDA_VISIBLE_DEVICES=4 python scripts/train_sepsis_benchmarks.py --task vr --model transformer --feature_subset full --epochs 30 --batch_size 64 --use_kgi --kgi_mode dgi'
CUDA_VISIBLE_DEVICES=4 python scripts/train_sepsis_benchmarks.py --task vr --model transformer --feature_subset full --epochs 30 --batch_size 64 --use_kgi --kgi_mode dgi
sleep 10

echo 'Launching Run 40/92: CUDA_VISIBLE_DEVICES=5 python scripts/train_sepsis_benchmarks.py --task vr --model transformer --feature_subset no_treatments --epochs 30 --batch_size 64'
CUDA_VISIBLE_DEVICES=5 python scripts/train_sepsis_benchmarks.py --task vr --model transformer --feature_subset no_treatments --epochs 30 --batch_size 64
sleep 10

echo 'Launching Run 41/92: CUDA_VISIBLE_DEVICES=4 python scripts/train_sepsis_benchmarks.py --task vr --model transformer --feature_subset no_treatments --epochs 30 --batch_size 64 --use_kgi --kgi_mode dki'
CUDA_VISIBLE_DEVICES=4 python scripts/train_sepsis_benchmarks.py --task vr --model transformer --feature_subset no_treatments --epochs 30 --batch_size 64 --use_kgi --kgi_mode dki
sleep 10

echo 'Launching Run 42/92: CUDA_VISIBLE_DEVICES=5 python scripts/train_sepsis_benchmarks.py --task vr --model transformer --feature_subset no_treatments --epochs 30 --batch_size 64 --use_kgi --kgi_mode dgi'
CUDA_VISIBLE_DEVICES=5 python scripts/train_sepsis_benchmarks.py --task vr --model transformer --feature_subset no_treatments --epochs 30 --batch_size 64 --use_kgi --kgi_mode dgi
sleep 10

echo 'Launching Run 43/92: CUDA_VISIBLE_DEVICES=4 python scripts/train_sepsis_benchmarks.py --task vr --model transformer --feature_subset core --epochs 30 --batch_size 64'
CUDA_VISIBLE_DEVICES=4 python scripts/train_sepsis_benchmarks.py --task vr --model transformer --feature_subset core --epochs 30 --batch_size 64
sleep 10

echo 'Launching Run 44/92: CUDA_VISIBLE_DEVICES=5 python scripts/train_sepsis_benchmarks.py --task vr --model transformer --feature_subset core --epochs 30 --batch_size 64 --use_kgi --kgi_mode dki'
CUDA_VISIBLE_DEVICES=5 python scripts/train_sepsis_benchmarks.py --task vr --model transformer --feature_subset core --epochs 30 --batch_size 64 --use_kgi --kgi_mode dki
sleep 10

echo 'Launching Run 45/92: CUDA_VISIBLE_DEVICES=4 python scripts/train_sepsis_benchmarks.py --task vr --model transformer --feature_subset core --epochs 30 --batch_size 64 --use_kgi --kgi_mode dgi'
CUDA_VISIBLE_DEVICES=4 python scripts/train_sepsis_benchmarks.py --task vr --model transformer --feature_subset core --epochs 30 --batch_size 64 --use_kgi --kgi_mode dgi
sleep 10

echo 'Launching Run 46/92: CUDA_VISIBLE_DEVICES=5 python scripts/train_sepsis_benchmarks.py --task vr --model transformer --feature_subset emergency --epochs 30 --batch_size 64'
CUDA_VISIBLE_DEVICES=5 python scripts/train_sepsis_benchmarks.py --task vr --model transformer --feature_subset emergency --epochs 30 --batch_size 64
sleep 10

echo 'Launching Run 47/92: CUDA_VISIBLE_DEVICES=4 python scripts/train_sepsis_benchmarks.py --task vr --model transformer --feature_subset emergency --epochs 30 --batch_size 64 --use_kgi --kgi_mode dki'
CUDA_VISIBLE_DEVICES=4 python scripts/train_sepsis_benchmarks.py --task vr --model transformer --feature_subset emergency --epochs 30 --batch_size 64 --use_kgi --kgi_mode dki
sleep 10

echo 'Launching Run 48/92: CUDA_VISIBLE_DEVICES=5 python scripts/train_sepsis_benchmarks.py --task vr --model transformer --feature_subset emergency --epochs 30 --batch_size 64 --use_kgi --kgi_mode dgi'
CUDA_VISIBLE_DEVICES=5 python scripts/train_sepsis_benchmarks.py --task vr --model transformer --feature_subset emergency --epochs 30 --batch_size 64 --use_kgi --kgi_mode dgi
sleep 10

echo 'Launching Run 49/92: CUDA_VISIBLE_DEVICES=6 python train.py model=joint model.task=ihm data.feature_subset=full model.imputator_name=saits trainer.max_epochs=30 model.imputator_kwargs.use_kgi=true model.imputator_kwargs.kgi_mode=dki'
CUDA_VISIBLE_DEVICES=6 python train.py model=joint model.task=ihm data.feature_subset=full model.imputator_name=saits trainer.max_epochs=30 model.imputator_kwargs.use_kgi=true model.imputator_kwargs.kgi_mode=dki
sleep 10

echo 'Launching Run 50/92: CUDA_VISIBLE_DEVICES=7 python train.py model=joint model.task=ihm data.feature_subset=full model.imputator_name=saits trainer.max_epochs=30 model.imputator_kwargs.use_kgi=true model.imputator_kwargs.kgi_mode=dgi'
CUDA_VISIBLE_DEVICES=7 python train.py model=joint model.task=ihm data.feature_subset=full model.imputator_name=saits trainer.max_epochs=30 model.imputator_kwargs.use_kgi=true model.imputator_kwargs.kgi_mode=dgi
sleep 10

echo 'Launching Run 51/92: CUDA_VISIBLE_DEVICES=6 python train.py model=joint model.task=ihm data.feature_subset=no_treatments model.imputator_name=saits trainer.max_epochs=30 model.imputator_kwargs.use_kgi=false'
CUDA_VISIBLE_DEVICES=6 python train.py model=joint model.task=ihm data.feature_subset=no_treatments model.imputator_name=saits trainer.max_epochs=30 model.imputator_kwargs.use_kgi=false
sleep 10

echo 'Launching Run 52/92: CUDA_VISIBLE_DEVICES=7 python train.py model=joint model.task=ihm data.feature_subset=no_treatments model.imputator_name=saits trainer.max_epochs=30 model.imputator_kwargs.use_kgi=true model.imputator_kwargs.kgi_mode=dki'
CUDA_VISIBLE_DEVICES=7 python train.py model=joint model.task=ihm data.feature_subset=no_treatments model.imputator_name=saits trainer.max_epochs=30 model.imputator_kwargs.use_kgi=true model.imputator_kwargs.kgi_mode=dki
sleep 10

echo 'Launching Run 53/92: CUDA_VISIBLE_DEVICES=6 python train.py model=joint model.task=ihm data.feature_subset=no_treatments model.imputator_name=saits trainer.max_epochs=30 model.imputator_kwargs.use_kgi=true model.imputator_kwargs.kgi_mode=dgi'
CUDA_VISIBLE_DEVICES=6 python train.py model=joint model.task=ihm data.feature_subset=no_treatments model.imputator_name=saits trainer.max_epochs=30 model.imputator_kwargs.use_kgi=true model.imputator_kwargs.kgi_mode=dgi
sleep 10

echo 'Launching Run 54/92: CUDA_VISIBLE_DEVICES=7 python train.py model=joint model.task=ihm data.feature_subset=core model.imputator_name=saits trainer.max_epochs=30 model.imputator_kwargs.use_kgi=false'
CUDA_VISIBLE_DEVICES=7 python train.py model=joint model.task=ihm data.feature_subset=core model.imputator_name=saits trainer.max_epochs=30 model.imputator_kwargs.use_kgi=false
sleep 10

echo 'Launching Run 55/92: CUDA_VISIBLE_DEVICES=6 python train.py model=joint model.task=ihm data.feature_subset=core model.imputator_name=saits trainer.max_epochs=30 model.imputator_kwargs.use_kgi=true model.imputator_kwargs.kgi_mode=dki'
CUDA_VISIBLE_DEVICES=6 python train.py model=joint model.task=ihm data.feature_subset=core model.imputator_name=saits trainer.max_epochs=30 model.imputator_kwargs.use_kgi=true model.imputator_kwargs.kgi_mode=dki
sleep 10

echo 'Launching Run 56/92: CUDA_VISIBLE_DEVICES=7 python train.py model=joint model.task=ihm data.feature_subset=core model.imputator_name=saits trainer.max_epochs=30 model.imputator_kwargs.use_kgi=true model.imputator_kwargs.kgi_mode=dgi'
CUDA_VISIBLE_DEVICES=7 python train.py model=joint model.task=ihm data.feature_subset=core model.imputator_name=saits trainer.max_epochs=30 model.imputator_kwargs.use_kgi=true model.imputator_kwargs.kgi_mode=dgi
sleep 10

echo 'Launching Run 57/92: CUDA_VISIBLE_DEVICES=6 python train.py model=joint model.task=ihm data.feature_subset=emergency model.imputator_name=saits trainer.max_epochs=30 model.imputator_kwargs.use_kgi=false'
CUDA_VISIBLE_DEVICES=6 python train.py model=joint model.task=ihm data.feature_subset=emergency model.imputator_name=saits trainer.max_epochs=30 model.imputator_kwargs.use_kgi=false
sleep 10

echo 'Launching Run 58/92: CUDA_VISIBLE_DEVICES=7 python train.py model=joint model.task=ihm data.feature_subset=emergency model.imputator_name=saits trainer.max_epochs=30 model.imputator_kwargs.use_kgi=true model.imputator_kwargs.kgi_mode=dki'
CUDA_VISIBLE_DEVICES=7 python train.py model=joint model.task=ihm data.feature_subset=emergency model.imputator_name=saits trainer.max_epochs=30 model.imputator_kwargs.use_kgi=true model.imputator_kwargs.kgi_mode=dki
sleep 10

echo 'Launching Run 59/92: CUDA_VISIBLE_DEVICES=6 python train.py model=joint model.task=ihm data.feature_subset=emergency model.imputator_name=saits trainer.max_epochs=30 model.imputator_kwargs.use_kgi=true model.imputator_kwargs.kgi_mode=dgi'
CUDA_VISIBLE_DEVICES=6 python train.py model=joint model.task=ihm data.feature_subset=emergency model.imputator_name=saits trainer.max_epochs=30 model.imputator_kwargs.use_kgi=true model.imputator_kwargs.kgi_mode=dgi
sleep 10

echo 'Launching Run 60/92: CUDA_VISIBLE_DEVICES=7 python train.py model=joint model.task=los data.feature_subset=full model.imputator_name=saits trainer.max_epochs=30 model.imputator_kwargs.use_kgi=true model.imputator_kwargs.kgi_mode=dki'
CUDA_VISIBLE_DEVICES=7 python train.py model=joint model.task=los data.feature_subset=full model.imputator_name=saits trainer.max_epochs=30 model.imputator_kwargs.use_kgi=true model.imputator_kwargs.kgi_mode=dki
sleep 10

echo 'Launching Run 61/92: CUDA_VISIBLE_DEVICES=6 python train.py model=joint model.task=los data.feature_subset=full model.imputator_name=saits trainer.max_epochs=30 model.imputator_kwargs.use_kgi=true model.imputator_kwargs.kgi_mode=dgi'
CUDA_VISIBLE_DEVICES=6 python train.py model=joint model.task=los data.feature_subset=full model.imputator_name=saits trainer.max_epochs=30 model.imputator_kwargs.use_kgi=true model.imputator_kwargs.kgi_mode=dgi
sleep 10

echo 'Launching Run 62/92: CUDA_VISIBLE_DEVICES=7 python train.py model=joint model.task=los data.feature_subset=no_treatments model.imputator_name=saits trainer.max_epochs=30 model.imputator_kwargs.use_kgi=false'
CUDA_VISIBLE_DEVICES=7 python train.py model=joint model.task=los data.feature_subset=no_treatments model.imputator_name=saits trainer.max_epochs=30 model.imputator_kwargs.use_kgi=false
sleep 10

echo 'Launching Run 63/92: CUDA_VISIBLE_DEVICES=6 python train.py model=joint model.task=los data.feature_subset=no_treatments model.imputator_name=saits trainer.max_epochs=30 model.imputator_kwargs.use_kgi=true model.imputator_kwargs.kgi_mode=dki'
CUDA_VISIBLE_DEVICES=6 python train.py model=joint model.task=los data.feature_subset=no_treatments model.imputator_name=saits trainer.max_epochs=30 model.imputator_kwargs.use_kgi=true model.imputator_kwargs.kgi_mode=dki
sleep 10

echo 'Launching Run 64/92: CUDA_VISIBLE_DEVICES=7 python train.py model=joint model.task=los data.feature_subset=no_treatments model.imputator_name=saits trainer.max_epochs=30 model.imputator_kwargs.use_kgi=true model.imputator_kwargs.kgi_mode=dgi'
CUDA_VISIBLE_DEVICES=7 python train.py model=joint model.task=los data.feature_subset=no_treatments model.imputator_name=saits trainer.max_epochs=30 model.imputator_kwargs.use_kgi=true model.imputator_kwargs.kgi_mode=dgi
sleep 10

echo 'Launching Run 65/92: CUDA_VISIBLE_DEVICES=6 python train.py model=joint model.task=los data.feature_subset=core model.imputator_name=saits trainer.max_epochs=30 model.imputator_kwargs.use_kgi=false'
CUDA_VISIBLE_DEVICES=6 python train.py model=joint model.task=los data.feature_subset=core model.imputator_name=saits trainer.max_epochs=30 model.imputator_kwargs.use_kgi=false
sleep 10

echo 'Launching Run 66/92: CUDA_VISIBLE_DEVICES=7 python train.py model=joint model.task=los data.feature_subset=core model.imputator_name=saits trainer.max_epochs=30 model.imputator_kwargs.use_kgi=true model.imputator_kwargs.kgi_mode=dki'
CUDA_VISIBLE_DEVICES=7 python train.py model=joint model.task=los data.feature_subset=core model.imputator_name=saits trainer.max_epochs=30 model.imputator_kwargs.use_kgi=true model.imputator_kwargs.kgi_mode=dki
sleep 10

echo 'Launching Run 67/92: CUDA_VISIBLE_DEVICES=6 python train.py model=joint model.task=los data.feature_subset=core model.imputator_name=saits trainer.max_epochs=30 model.imputator_kwargs.use_kgi=true model.imputator_kwargs.kgi_mode=dgi'
CUDA_VISIBLE_DEVICES=6 python train.py model=joint model.task=los data.feature_subset=core model.imputator_name=saits trainer.max_epochs=30 model.imputator_kwargs.use_kgi=true model.imputator_kwargs.kgi_mode=dgi
sleep 10

echo 'Launching Run 68/92: CUDA_VISIBLE_DEVICES=7 python train.py model=joint model.task=los data.feature_subset=emergency model.imputator_name=saits trainer.max_epochs=30 model.imputator_kwargs.use_kgi=false'
CUDA_VISIBLE_DEVICES=7 python train.py model=joint model.task=los data.feature_subset=emergency model.imputator_name=saits trainer.max_epochs=30 model.imputator_kwargs.use_kgi=false
sleep 10

echo 'Launching Run 69/92: CUDA_VISIBLE_DEVICES=6 python train.py model=joint model.task=los data.feature_subset=emergency model.imputator_name=saits trainer.max_epochs=30 model.imputator_kwargs.use_kgi=true model.imputator_kwargs.kgi_mode=dki'
CUDA_VISIBLE_DEVICES=6 python train.py model=joint model.task=los data.feature_subset=emergency model.imputator_name=saits trainer.max_epochs=30 model.imputator_kwargs.use_kgi=true model.imputator_kwargs.kgi_mode=dki
sleep 10

echo 'Launching Run 70/92: CUDA_VISIBLE_DEVICES=7 python train.py model=joint model.task=los data.feature_subset=emergency model.imputator_name=saits trainer.max_epochs=30 model.imputator_kwargs.use_kgi=true model.imputator_kwargs.kgi_mode=dgi'
CUDA_VISIBLE_DEVICES=7 python train.py model=joint model.task=los data.feature_subset=emergency model.imputator_name=saits trainer.max_epochs=30 model.imputator_kwargs.use_kgi=true model.imputator_kwargs.kgi_mode=dgi
sleep 10

echo 'Launching Run 71/92: CUDA_VISIBLE_DEVICES=6 python train.py model=joint model.task=ss data.feature_subset=full model.imputator_name=saits trainer.max_epochs=30 model.imputator_kwargs.use_kgi=true model.imputator_kwargs.kgi_mode=dki'
CUDA_VISIBLE_DEVICES=6 python train.py model=joint model.task=ss data.feature_subset=full model.imputator_name=saits trainer.max_epochs=30 model.imputator_kwargs.use_kgi=true model.imputator_kwargs.kgi_mode=dki
sleep 10

echo 'Launching Run 72/92: CUDA_VISIBLE_DEVICES=7 python train.py model=joint model.task=ss data.feature_subset=full model.imputator_name=saits trainer.max_epochs=30 model.imputator_kwargs.use_kgi=true model.imputator_kwargs.kgi_mode=dgi'
CUDA_VISIBLE_DEVICES=7 python train.py model=joint model.task=ss data.feature_subset=full model.imputator_name=saits trainer.max_epochs=30 model.imputator_kwargs.use_kgi=true model.imputator_kwargs.kgi_mode=dgi
sleep 10

echo 'Launching Run 73/92: CUDA_VISIBLE_DEVICES=6 python train.py model=joint model.task=ss data.feature_subset=no_treatments model.imputator_name=saits trainer.max_epochs=30 model.imputator_kwargs.use_kgi=false'
CUDA_VISIBLE_DEVICES=6 python train.py model=joint model.task=ss data.feature_subset=no_treatments model.imputator_name=saits trainer.max_epochs=30 model.imputator_kwargs.use_kgi=false
sleep 10

echo 'Launching Run 74/92: CUDA_VISIBLE_DEVICES=7 python train.py model=joint model.task=ss data.feature_subset=no_treatments model.imputator_name=saits trainer.max_epochs=30 model.imputator_kwargs.use_kgi=true model.imputator_kwargs.kgi_mode=dki'
CUDA_VISIBLE_DEVICES=7 python train.py model=joint model.task=ss data.feature_subset=no_treatments model.imputator_name=saits trainer.max_epochs=30 model.imputator_kwargs.use_kgi=true model.imputator_kwargs.kgi_mode=dki
sleep 10

echo 'Launching Run 75/92: CUDA_VISIBLE_DEVICES=6 python train.py model=joint model.task=ss data.feature_subset=no_treatments model.imputator_name=saits trainer.max_epochs=30 model.imputator_kwargs.use_kgi=true model.imputator_kwargs.kgi_mode=dgi'
CUDA_VISIBLE_DEVICES=6 python train.py model=joint model.task=ss data.feature_subset=no_treatments model.imputator_name=saits trainer.max_epochs=30 model.imputator_kwargs.use_kgi=true model.imputator_kwargs.kgi_mode=dgi
sleep 10

echo 'Launching Run 76/92: CUDA_VISIBLE_DEVICES=7 python train.py model=joint model.task=ss data.feature_subset=core model.imputator_name=saits trainer.max_epochs=30 model.imputator_kwargs.use_kgi=false'
CUDA_VISIBLE_DEVICES=7 python train.py model=joint model.task=ss data.feature_subset=core model.imputator_name=saits trainer.max_epochs=30 model.imputator_kwargs.use_kgi=false
sleep 10

echo 'Launching Run 77/92: CUDA_VISIBLE_DEVICES=6 python train.py model=joint model.task=ss data.feature_subset=core model.imputator_name=saits trainer.max_epochs=30 model.imputator_kwargs.use_kgi=true model.imputator_kwargs.kgi_mode=dki'
CUDA_VISIBLE_DEVICES=6 python train.py model=joint model.task=ss data.feature_subset=core model.imputator_name=saits trainer.max_epochs=30 model.imputator_kwargs.use_kgi=true model.imputator_kwargs.kgi_mode=dki
sleep 10

echo 'Launching Run 78/92: CUDA_VISIBLE_DEVICES=7 python train.py model=joint model.task=ss data.feature_subset=core model.imputator_name=saits trainer.max_epochs=30 model.imputator_kwargs.use_kgi=true model.imputator_kwargs.kgi_mode=dgi'
CUDA_VISIBLE_DEVICES=7 python train.py model=joint model.task=ss data.feature_subset=core model.imputator_name=saits trainer.max_epochs=30 model.imputator_kwargs.use_kgi=true model.imputator_kwargs.kgi_mode=dgi
sleep 10

echo 'Launching Run 79/92: CUDA_VISIBLE_DEVICES=6 python train.py model=joint model.task=ss data.feature_subset=emergency model.imputator_name=saits trainer.max_epochs=30 model.imputator_kwargs.use_kgi=false'
CUDA_VISIBLE_DEVICES=6 python train.py model=joint model.task=ss data.feature_subset=emergency model.imputator_name=saits trainer.max_epochs=30 model.imputator_kwargs.use_kgi=false
sleep 10

echo 'Launching Run 80/92: CUDA_VISIBLE_DEVICES=7 python train.py model=joint model.task=ss data.feature_subset=emergency model.imputator_name=saits trainer.max_epochs=30 model.imputator_kwargs.use_kgi=true model.imputator_kwargs.kgi_mode=dki'
CUDA_VISIBLE_DEVICES=7 python train.py model=joint model.task=ss data.feature_subset=emergency model.imputator_name=saits trainer.max_epochs=30 model.imputator_kwargs.use_kgi=true model.imputator_kwargs.kgi_mode=dki
sleep 10

echo 'Launching Run 81/92: CUDA_VISIBLE_DEVICES=6 python train.py model=joint model.task=ss data.feature_subset=emergency model.imputator_name=saits trainer.max_epochs=30 model.imputator_kwargs.use_kgi=true model.imputator_kwargs.kgi_mode=dgi'
CUDA_VISIBLE_DEVICES=6 python train.py model=joint model.task=ss data.feature_subset=emergency model.imputator_name=saits trainer.max_epochs=30 model.imputator_kwargs.use_kgi=true model.imputator_kwargs.kgi_mode=dgi
sleep 10

echo 'Launching Run 82/92: CUDA_VISIBLE_DEVICES=7 python train.py model=joint model.task=vr data.feature_subset=full model.imputator_name=saits trainer.max_epochs=30 model.imputator_kwargs.use_kgi=true model.imputator_kwargs.kgi_mode=dki'
CUDA_VISIBLE_DEVICES=7 python train.py model=joint model.task=vr data.feature_subset=full model.imputator_name=saits trainer.max_epochs=30 model.imputator_kwargs.use_kgi=true model.imputator_kwargs.kgi_mode=dki
sleep 10

echo 'Launching Run 83/92: CUDA_VISIBLE_DEVICES=6 python train.py model=joint model.task=vr data.feature_subset=full model.imputator_name=saits trainer.max_epochs=30 model.imputator_kwargs.use_kgi=true model.imputator_kwargs.kgi_mode=dgi'
CUDA_VISIBLE_DEVICES=6 python train.py model=joint model.task=vr data.feature_subset=full model.imputator_name=saits trainer.max_epochs=30 model.imputator_kwargs.use_kgi=true model.imputator_kwargs.kgi_mode=dgi
sleep 10

echo 'Launching Run 84/92: CUDA_VISIBLE_DEVICES=7 python train.py model=joint model.task=vr data.feature_subset=no_treatments model.imputator_name=saits trainer.max_epochs=30 model.imputator_kwargs.use_kgi=false'
CUDA_VISIBLE_DEVICES=7 python train.py model=joint model.task=vr data.feature_subset=no_treatments model.imputator_name=saits trainer.max_epochs=30 model.imputator_kwargs.use_kgi=false
sleep 10

echo 'Launching Run 85/92: CUDA_VISIBLE_DEVICES=6 python train.py model=joint model.task=vr data.feature_subset=no_treatments model.imputator_name=saits trainer.max_epochs=30 model.imputator_kwargs.use_kgi=true model.imputator_kwargs.kgi_mode=dki'
CUDA_VISIBLE_DEVICES=6 python train.py model=joint model.task=vr data.feature_subset=no_treatments model.imputator_name=saits trainer.max_epochs=30 model.imputator_kwargs.use_kgi=true model.imputator_kwargs.kgi_mode=dki
sleep 10

echo 'Launching Run 86/92: CUDA_VISIBLE_DEVICES=7 python train.py model=joint model.task=vr data.feature_subset=no_treatments model.imputator_name=saits trainer.max_epochs=30 model.imputator_kwargs.use_kgi=true model.imputator_kwargs.kgi_mode=dgi'
CUDA_VISIBLE_DEVICES=7 python train.py model=joint model.task=vr data.feature_subset=no_treatments model.imputator_name=saits trainer.max_epochs=30 model.imputator_kwargs.use_kgi=true model.imputator_kwargs.kgi_mode=dgi
sleep 10

echo 'Launching Run 87/92: CUDA_VISIBLE_DEVICES=6 python train.py model=joint model.task=vr data.feature_subset=core model.imputator_name=saits trainer.max_epochs=30 model.imputator_kwargs.use_kgi=false'
CUDA_VISIBLE_DEVICES=6 python train.py model=joint model.task=vr data.feature_subset=core model.imputator_name=saits trainer.max_epochs=30 model.imputator_kwargs.use_kgi=false
sleep 10

echo 'Launching Run 88/92: CUDA_VISIBLE_DEVICES=7 python train.py model=joint model.task=vr data.feature_subset=core model.imputator_name=saits trainer.max_epochs=30 model.imputator_kwargs.use_kgi=true model.imputator_kwargs.kgi_mode=dki'
CUDA_VISIBLE_DEVICES=7 python train.py model=joint model.task=vr data.feature_subset=core model.imputator_name=saits trainer.max_epochs=30 model.imputator_kwargs.use_kgi=true model.imputator_kwargs.kgi_mode=dki
sleep 10

echo 'Launching Run 89/92: CUDA_VISIBLE_DEVICES=6 python train.py model=joint model.task=vr data.feature_subset=core model.imputator_name=saits trainer.max_epochs=30 model.imputator_kwargs.use_kgi=true model.imputator_kwargs.kgi_mode=dgi'
CUDA_VISIBLE_DEVICES=6 python train.py model=joint model.task=vr data.feature_subset=core model.imputator_name=saits trainer.max_epochs=30 model.imputator_kwargs.use_kgi=true model.imputator_kwargs.kgi_mode=dgi
sleep 10

echo 'Launching Run 90/92: CUDA_VISIBLE_DEVICES=7 python train.py model=joint model.task=vr data.feature_subset=emergency model.imputator_name=saits trainer.max_epochs=30 model.imputator_kwargs.use_kgi=false'
CUDA_VISIBLE_DEVICES=7 python train.py model=joint model.task=vr data.feature_subset=emergency model.imputator_name=saits trainer.max_epochs=30 model.imputator_kwargs.use_kgi=false
sleep 10

echo 'Launching Run 91/92: CUDA_VISIBLE_DEVICES=6 python train.py model=joint model.task=vr data.feature_subset=emergency model.imputator_name=saits trainer.max_epochs=30 model.imputator_kwargs.use_kgi=true model.imputator_kwargs.kgi_mode=dki'
CUDA_VISIBLE_DEVICES=6 python train.py model=joint model.task=vr data.feature_subset=emergency model.imputator_name=saits trainer.max_epochs=30 model.imputator_kwargs.use_kgi=true model.imputator_kwargs.kgi_mode=dki
sleep 10

echo 'Launching Run 92/92: CUDA_VISIBLE_DEVICES=7 python train.py model=joint model.task=vr data.feature_subset=emergency model.imputator_name=saits trainer.max_epochs=30 model.imputator_kwargs.use_kgi=true model.imputator_kwargs.kgi_mode=dgi'
CUDA_VISIBLE_DEVICES=7 python train.py model=joint model.task=vr data.feature_subset=emergency model.imputator_name=saits trainer.max_epochs=30 model.imputator_kwargs.use_kgi=true model.imputator_kwargs.kgi_mode=dgi
sleep 10

