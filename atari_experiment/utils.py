import os

exp_log = "logs/"

def gen_exp_log_dir_name(target_task_id, base_task_id=None):
    target_task_id_name = target_task_id.replace("/", "-")
    if base_task_id:
        base_task_id_name = base_task_id.replace("/", "-")
        exp_log_prefix = os.path.join(
            exp_log,
            f"ppo2_{base_task_id_name}_to_{target_task_id_name}")
    else:
        exp_log_prefix = os.path.join(exp_log, f"ppo2_{target_task_id_name}")

    i = 0
    while os.path.isdir(f"{exp_log_prefix}_{i}"):
        i += 1
    exp_log_path = f"{exp_log_prefix}_{i}"
    os.mkdir(exp_log_path)
    tensorboard_log = os.path.join(exp_log_path, f"tensorboard_log")
    model_save_path = os.path.join(exp_log_path, f"trained_model.zip")

    return exp_log_path, tensorboard_log, model_save_path
