hyperparams = {
    "Reacher-v4": dict(
        normalize=False,
        n_envs=1,
        policy="MlpPolicy",
        n_timesteps=100000.0,
        batch_size=32,
        n_steps=512,
        gamma=0.9,
        learning_rate=0.000104019,
        ent_coef=7.52585e-08,
        clip_range=0.3,
        n_epochs=5,
        gae_lambda=1.0,
        max_grad_norm=0.9,
        vf_coef=0.950368
    )
}