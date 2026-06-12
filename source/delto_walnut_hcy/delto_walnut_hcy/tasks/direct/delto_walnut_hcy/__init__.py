import gymnasium as gym

from . import agents

gym.register(
    id="Template-Delto-Walnut-Direct-v0",
    entry_point=f"{__name__}.delto_walnut_hcy_env:DeltoWalnutEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.delto_walnut_hcy_env_cfg:DeltoWalnutEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)
