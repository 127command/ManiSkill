python ppo.py --env_id="final" --num_envs=512 --update_epochs=8 --num_minibatches=32  --total_timesteps=2_000_00000 --eval_freq=10 --num-steps=500 --num_eval_steps=500 --track --eval-reconfiguration-freq=5
python sac.py --env_id="final" --num_envs=256 --batch-size=32 --total_timesteps=2_000_00000 --eval_freq=500

python ppo_fast.py --env_id="final" --num_envs=512 --update_epochs=8 --num_minibatches=32  --total_timesteps=2_000_00000 --eval_freq=10 --num-steps=500 --num_eval_steps=500 --track

python ppo.py --env_id="final" --num_envs=512 --update_epochs=8 --num_minibatches=32  --total_timesteps=2_000_00000 --eval_freq=10 --num-steps=500 --num_eval_steps=500 --track --checkpoint /home/changruinian/bowen/EAI/ManiSkill/examples/baselines/ppo/runs/final__ppo__1__1735267212/ckpt_231.pt

python ppo.py --env_id="final" --num_envs=512 --update_epochs=8 --num_minibatches=32  --total_timesteps=2_000_00000 --eval_freq=10 --num-steps=500 --num_eval_steps=500 --track --checkpoint /home/changruinian/bowen/EAI/ManiSkill/examples/baselines/ppo/runs/final__ppo__1__1735390761/ckpt_781.pt