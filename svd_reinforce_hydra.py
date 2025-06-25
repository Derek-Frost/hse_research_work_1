import gc
import json
import os
from datetime import datetime

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

from base_model.tinyllamainstruct import LocalTinyLlamaModel  # Используем локальную модель
from utils import (
    eval_model,
    eval_model_experts_prompt_based,
    forward,
    load_local_safetensors,
    load_hf_params_to_vllm
)
from logging_utils import Metrics, get_mean_std_max_min_dict
from optim_modules import OptimizationAlgorithm
from policy import Policy
from tasks import Task


def wandb_init(cfg, run_name: str, group_name: str, log_dir: str):
    import wandb

    config_dict = OmegaConf.to_container(
        cfg,
        resolve=True,
        throw_on_missing=False,
    )
    config_dict["log_dir"] = log_dir
    config_dict["wandb_run_name"] = run_name
    config_dict["wandb_group_name"] = group_name

    # wandb имеет ограничение на длину group name – 128 символов
    wandb.init(
        project=cfg.wandb_project,
        group=group_name[:127],
        name=run_name[:127],
        config=config_dict,
    )
    return wandb


@hydra.main(version_base=None, config_path="cfgs", config_name="config")
def main(cfg):
    """Main function."""

    num_iters = cfg.num_iters
    test_interval = cfg.test_interval

    batch_size = cfg.batch_size
    seed = cfg.seed
    policy_name = cfg.policy_name
    test_only = cfg.test_only
    save_legacy_params = cfg.save_legacy_params
    exp_name = cfg.exp_name
    run_name = cfg.run_name

    task_name = cfg.task_name

    load_ckpt = cfg.load_ckpt
    use_lora = cfg.use_lora
    prompt_based_eval = cfg.prompt_based_eval
    experts_path_dict = cfg.experts_path_dict
    print(f"load_ckpt=={load_ckpt}")
    resuming_from_ckpt = False
    if load_ckpt is not None:
        if load_ckpt == "scratch" or load_ckpt == "base":
            resuming_from_ckpt = False
        else:
            resuming_from_ckpt = True

    # Создаём загрузчик задачи
    task_loader: Task = hydra.utils.instantiate(cfg.task_loader)

    # Инстанцируем локальную модель (она возвращает путь к файлу safetensors)
    base_model_obj: LocalTinyLlamaModel = hydra.utils.instantiate(cfg.base_model)
    # model_id теперь == './models' (папка)
    model_id = base_model_obj.get_model_id()
    # param_file == './models/TinyLlama-1.1B.safetensors'
    param_file = base_model_obj.get_param_file()
    # Папка, которая содержит config.json, tokenizer.json и т.д.
    model_dir = model_id
    config_path = os.path.join(model_dir, "config.json")
    config = AutoConfig.from_pretrained(config_path)
  
    decomposed_param_file = base_model_obj.get_dec_param_file()


    extract_svd = cfg.extract_svd or (not os.path.exists(decomposed_param_file))

    has_training_split = task_loader.has_training_split
    has_transfer_split = task_loader.has_transfer_split

    if not has_training_split:
        assert test_only, "Cannot train on a task with no training split"

    if exp_name is None:
        exp_name = "temp"

    metrics_to_log = Metrics()

    # Создаём директорию для логов
    if run_name is None:
        now = datetime.now()
        run_name = now.strftime("%Y%m%d-%H%M%S")
    if test_only and (not resuming_from_ckpt):
        log_dir = f"{cfg.out_dir}/{task_name}/{cfg.base_model_name}_base"
        group_name = cfg.base_model_name
    else:
        log_dir = f"{cfg.out_dir}/{task_name}/{policy_name}/{exp_name}/{run_name}"
        group_name = cfg.wandb_group_name
    os.makedirs(log_dir, exist_ok=True)

    # Получаем vllm модель через task_loader
    vllm_model = task_loader.get_vllm_model(model_id=model_id)

    train_eval, *test_evals = task_loader.get_evaluator()
    if task_loader.has_transfer_split:
        test_eval, transfer_eval = test_evals
    else:
        test_eval = test_evals[0]

    train_data, train_ix, valid_ix = task_loader.get_train_data()
    gpu = torch.device("cuda:1")
    np_random = np.random.RandomState(seed)

    # Загрузка модели через локальный safetensors
    # Предполагается, что в той же директории, что и safetensors-файл, находится config.json
    config = AutoConfig.from_pretrained(model_id) 

    # Выбираем устройство и тип тензора
    if extract_svd:
        device = torch.device("cpu")
        torch_dtype = torch.float32
    else:
        device = gpu
        torch_dtype = torch.float16

    #model = AutoModelForCausalLM.from_config(config)
    #model.to(device)

    model = AutoModelForCausalLM.from_config(config).to(device).to(dtype=torch_dtype)

    # Загружаем веса из локального safetensors-файла
    load_local_safetensors(model, param_file)

    # Загружаем токенайзер из директории модели
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    base_params = model.state_dict()
    for k in base_params:
        base_params[k] = base_params[k].to(torch.float16)
    original_model_params = {
        k: v.clone().detach().cpu() for k, v in base_params.items() if "mlp" in k
    }

    # Загрузка декомпозированных параметров
    if not os.path.exists(decomposed_param_file):
        print("Decomposed params not found. Decomposing...")
        decomposed_params = {}
        for k, v in base_params.items():
            if "norm" not in k:
                print(k)
                v_fp32 = v.to(torch.float32).cpu()
                U, S, V = torch.svd(v_fp32)
                decomposed_params[f"{k}.U"] = U.to(torch.float16)
                decomposed_params[f"{k}.S"] = S.to(torch.float16)
                decomposed_params[f"{k}.V"] = V.to(torch.float16)
        torch.save(decomposed_params, decomposed_param_file)
        print("Successfully decomposed model - returning")
        return
    elif extract_svd:
        print(f"ERROR: SVD file already exists at {decomposed_param_file}")
    else:
        print("Decomposed params found. Loading...")
        assert not extract_svd
        decomposed_params = torch.load(decomposed_param_file)
    for k, v in decomposed_params.items():
        decomposed_params[k] = v.to(torch.float16).to(gpu)

    if cfg.wandb_log:
        wandb = wandb_init(
            cfg=cfg, group_name=group_name, run_name=run_name, log_dir=log_dir
        )

    policy: Policy = hydra.utils.instantiate(
        cfg.shakeoff_policy,
        base_params=base_params,
        decomposed_params=decomposed_params,
        gpu=gpu,
    )

    # float32 -> float16
    learnable_params = policy.get_learnable_params()
    for k, v in learnable_params.items():
        learnable_params[k] = v.to(torch.float16).to(gpu)


    optimization_algorithm: OptimizationAlgorithm = hydra.utils.instantiate(
        cfg.optimization_algorithm,
        policy=policy,
        gpu=gpu,
    )

    if resuming_from_ckpt and os.path.exists(load_ckpt):
        print(f"Starting from checkpoint at: {load_ckpt}")
        # загрузка lora весов
        if use_lora:
            assert os.path.isdir(load_ckpt), "ckpt for lora must be dir to lora adapter"
            from peft import PeftModel

            lora_model = PeftModel.from_pretrained(model, load_ckpt)
            merged_model = lora_model.merge_and_unload()
            new_params = merged_model.state_dict()
        # загрузка svd expert
        elif "learnable_params" in load_ckpt:
            learnable_params = torch.load(load_ckpt)
            for k, v in learnable_params.items():
                learnable_params[k] = v.to(torch.float16).to(gpu)
            assert test_only
            new_params = forward(
                policy, model, base_params, decomposed_params, learnable_params
            )
        else:
            state_dict = torch.load(load_ckpt, weights_only=True)
            policy.load_state_dict(state_dict=state_dict)
            if test_only:
                learnable_params = policy.get_learnable_params()
            new_params = forward(
                policy, model, base_params, decomposed_params, learnable_params
            )
        from utils import load_hf_params_to_vllm
        load_hf_params_to_vllm(new_params, vllm_model.llm)
    else:
        print(f"Starting from the base model as load_ckpt=={load_ckpt} and resuming_from_ckpt=={resuming_from_ckpt} and experts_path_dict=={experts_path_dict}")

    model.eval()

    # Оценка с использованием prompt-based метода и классификации.
    if test_only and prompt_based_eval:
        test_data_dict = eval_model_experts_prompt_based(
            vllm_model,
            test_eval,
            experts_path_dict,
            policy,
            model,
            base_params,
            decomposed_params,
            task_loader.target_metric_test,
        )
        test_data_dict["type"] = "test"
        if cfg.wandb_log:
            wandb.log(test_data_dict)
        with open(f"{log_dir}/eval_results.json", "w") as f:
            json.dump(test_data_dict, f, indent=4)
        print(f"Test evaluation results: {test_data_dict}")

        if has_transfer_split:
            transfer_data_dict = eval_model_experts_prompt_based(
                vllm_model,
                transfer_eval,
                experts_path_dict,
                policy,
                model,
                base_params,
                decomposed_params,
                task_loader.target_metric_transfer,
            )
            transfer_data_dict["type"] = "transfer"
            if cfg.wandb_log:
                wandb.log(transfer_data_dict)
            with open(f"{log_dir}/eval_results.json", "w") as f:
                json.dump(transfer_data_dict, f, indent=4)
            print(f"Transfer evaluation results: {transfer_data_dict}")

        return

    # Нейадаптивная оценка на train, val, test.
    if test_only and not prompt_based_eval:
        data_dict = {}
        details_dict = {}
        if has_training_split:
            train_res = eval_model(vllm_model, train_eval, train_ix)
            valid_res = eval_model(vllm_model, train_eval, valid_ix)
            data_dict["train_acc"] = train_res.aggregate_metrics[
                task_loader.target_metric_train
            ]
            data_dict["valid_acc"] = valid_res.aggregate_metrics[
                task_loader.target_metric_valid
            ]
            details_dict["train"] = train_res.sample_details
            details_dict["valid"] = valid_res.sample_details
        test_res = eval_model(vllm_model, test_eval)
        data_dict["test_acc"] = test_res.aggregate_metrics[
            task_loader.target_metric_test
        ]
        details_dict["test"] = test_res.sample_details
        if has_transfer_split:
            transfer_res = eval_model(vllm_model, transfer_eval)
            data_dict["transfer_acc"] = transfer_res.aggregate_metrics[
                task_loader.target_metric_transfer
            ]
            details_dict["transfer"] = transfer_res.sample_details
        if cfg.wandb_log:
            wandb.log(data_dict)
        with open(f"{log_dir}/eval_results.json", "w") as f:
            json.dump(data_dict, f, indent=4)
        print(f"Evaluation results: {data_dict}")
        return

    learnable_params = policy.get_learnable_params()
    for k, v in learnable_params.items():
        learnable_params[k] = v.to(torch.float16).to(gpu)

    for k in learnable_params:
        model.get_parameter(k).requires_grad_(True)

    # Основной цикл обучения.
    if batch_size is None:
        clipped_batch_size = len(list(train_ix))
    else:
        clipped_batch_size = min(batch_size, len(list(train_ix)))
    best_val_acc = 0.0
    test_at_best = 0.0
    transfer_at_best = 0.0
    for i in range(num_iters):

        batch_ix = np_random.choice(train_ix, size=clipped_batch_size, replace=False)

        optimization_algorithm.step_optimization(
            model_id=model_id,
            model=model,
            tokenizer=tokenizer,
            policy=policy,
            task_loader=task_loader,
            batch_ix=batch_ix,
            train_data=train_data,
            train_eval=train_eval,
            base_params=base_params,
            decomposed_params=decomposed_params,
            original_model_params=original_model_params,
            metrics_to_log=metrics_to_log,
            vllm_model=vllm_model,
        )

        with torch.no_grad():
            lists_to_log = {}
            grads = [p.grad for p in policy.trainable_params]
            if grads[0] is not None:
                grad_mean = [g.mean().item() for g in grads]
                grad_mags = [torch.linalg.vector_norm(g).item() for g in grads]
                lists_to_log["grad_mean"] = grad_mean
                lists_to_log["grad_mags"] = grad_mags

                param_mags = [
                    torch.linalg.vector_norm(p).item() for p in policy.trainable_params
                ]
                lists_to_log["policy_param_mag"] = param_mags

            generated_params_list = list(learnable_params.values())
            generated_param_mean = [p.mean().item() for p in generated_params_list]
            generated_param_mags = [
                torch.linalg.vector_norm(p).item() for p in generated_params_list
            ]
            lists_to_log["generated_param_mean"] = generated_param_mean
            lists_to_log["generated_param_mags"] = generated_param_mags

            list_stats = {}
            for k, v in lists_to_log.items():
                list_stats.update(get_mean_std_max_min_dict(array=v, prefix=k))
            metrics_to_log.update(**list_stats)

        optimization_algorithm.update(policy=policy)

        gc.collect()
        torch.cuda.empty_cache()
        model.zero_grad()

        value_mean = list_stats.get("generated_param_mean/mean", None)
        grad_mean_mag = list_stats.get("grad_mags/mean", None)
        print(
            f"Iter {i}: "
            + f"param_mean={value_mean}, "
            + f"grad_mean_mag={grad_mean_mag}"
        )
        optimization_algorithm.log_optim(metrics_to_log=metrics_to_log)

        if i % test_interval == 0:
            
            learnable_params = policy.get_learnable_params()
            for k, v in learnable_params.items():
                learnable_params[k] = v.to(torch.float16).to(gpu)

            forward(policy, model, base_params, decomposed_params, learnable_params)
            from utils import load_hf_params_to_vllm
            load_hf_params_to_vllm(model.state_dict(), vllm_model.llm)

            train_res = eval_model(vllm_model, train_eval, train_ix)
            valid_res = eval_model(vllm_model, train_eval, valid_ix)
            test_res = eval_model(vllm_model, test_eval)
            if has_transfer_split:
                transfer_res = eval_model(vllm_model, transfer_eval)
            if (
                valid_res.aggregate_metrics[task_loader.target_metric_valid]
                > best_val_acc
            ):
                best_val_acc = valid_res.aggregate_metrics[
                    task_loader.target_metric_valid
                ]
                test_at_best = test_res.aggregate_metrics[
                    task_loader.target_metric_test
                ]
                if has_transfer_split:
                    transfer_at_best = transfer_res.aggregate_metrics[
                        task_loader.target_metric_transfer
                    ]
                print("best_val_acc updated")
                path = f"{log_dir}/policy_params.pt"
                torch.save(policy.state_dict(), path)
                if save_legacy_params:
                    torch.save(learnable_params, f"{log_dir}/learnable_params.pt")

            path = f"{log_dir}/policy_params_latest.pt"
            torch.save(policy.state_dict(), path)
            if save_legacy_params:
                torch.save(learnable_params, f"{log_dir}/learnable_params_latest.pt")

            policy.record_state(metrics_to_log=metrics_to_log)
            data_dict = {
                "iter": i,
                "best_val_acc": best_val_acc,
                "test_at_best_val": test_at_best,
                "train_acc": train_res.aggregate_metrics[
                    task_loader.target_metric_train
                ],
                "valid_acc": valid_res.aggregate_metrics[
                    task_loader.target_metric_valid
                ],
                "test_acc": test_res.aggregate_metrics[task_loader.target_metric_test],
                **metrics_to_log.get(),
            }
            if has_transfer_split:
                data_dict["transfer_acc"] = transfer_res.aggregate_metrics[
                    task_loader.target_metric_transfer
                ]
                data_dict["transfer_at_best_val"] = transfer_at_best
            if cfg.wandb_log:
                wandb.log(data_dict)
            with open(f"{log_dir}/reinforce_log.json", "a") as f:
                json_data = json.dumps(data_dict, indent=4)
                f.write(json_data)
                f.write("\n")
            metrics_to_log.reset()


if __name__ == "__main__":
    main()
