exp_name=0821_bipedal_walker_hardcore_random_context
mkdir $exp_name
cd $exp_name
nohup srun -p chpc --exclusive --gres=gpu:4 \
  python ~/td3-context/run_experiment.py \
  --env-name BipedalWalkerHardcore-v3 \
  --context-mode "random" \
  >../$exp_name.log 2>&1 &
cd ..
sleep 5

exp_name=0821_bipedal_walker_hardcore_no_context
mkdir $exp_name
cd $exp_name
nohup srun -p chpc --exclusive --gres=gpu:4 \
  python ~/td3-context/run_experiment.py \
  --env-name BipedalWalkerHardcore-v3 \
  --context-mode "disable" \
  >../$exp_name.log 2>&1 &
cd ..
sleep 5

exp_name=0821_bipedal_walker_random_context
mkdir $exp_name
cd $exp_name
nohup srun -p chpc --exclusive --gres=gpu:4 \
  python ~/td3-context/run_experiment.py \
  --env-name BipedalWalker-v3 \
  --context-mode "random" \
  >../$exp_name.log 2>&1 &
cd ..
sleep 5

exp_name=0821_bipedal_walker_no_context
mkdir $exp_name
cd $exp_name
nohup srun -p chpc --exclusive --gres=gpu:4 \
  python ~/td3-context/run_experiment.py \
  --env-name BipedalWalker-v3 \
  --context-mode "disable" \
  >../$exp_name.log 2>&1 &
cd ..
sleep 5
