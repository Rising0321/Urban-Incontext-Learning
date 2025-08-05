import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.MDT import MaskFlowDiffusionGPT, GPTConfig
from data.datasets import DatasetMAE
from utils.utils import set_random_seed, calc, parse_args, process_outliers
from tqdm import tqdm, trange
import random
from diffusers import DPMSolverMultistepScheduler, DDPMScheduler
import swanlab
from swanlab import Settings
from transformers import get_cosine_schedule_with_warmup
from copy import deepcopy
import math

dir = "/home/wangb/STIC"

cities = ["Chicago", "Manhattan"]

targets = ["carbon", "house", "crash"]

tasks = ["carbon", "house", "crash"]


class DummyRun:
    def log(self, *args, **kwargs):
        pass

    def finish(self):
        pass


def normalize(x):
    x = np.array(x, dtype=np.float64)

    mean = np.mean(x)
    std = np.std(x)
    if std < 1e-8:
        return np.zeros_like(x), {"mean": mean, "std": std, "min": 0, "max": 0, "valid": False}

    standardized = (x - mean) / std
    # standardized = np.clip(standardized, -3, 3)
    min_val, max_val = standardized.min(), standardized.max()

    return standardized, {"mean": mean, "std": std, "min": min_val, "max": max_val, "valid": True}


task_norm_stats = {}


def denormalize(x_norm, stats):
    if not stats.get("valid", True):
        return np.zeros_like(x_norm)

    min_val, max_val = stats["min"], stats["max"]
    mean, std = stats["mean"], stats["std"]

    standardized = (x_norm + 1) * 0.5 * (max_val - min_val) + min_val
    return standardized

EMBEDDING_PATH = "./embeddings"
DATA_PATH = "./data"

def create_datasets(args):
    # 训练集：80% POI Taxi
    # 验证集：POI的剩下的 Taxi剩下的
    # 测试集：1 个POI， 1个 Taxi，全部的 Crime, Population, Carbon

    embedding = np.load(f"{EMBEDDING_PATH}/{args.source}/{args.dataset}.npy")
    print(embedding.shape)

    train_datas = []
    test_datas = []

    for target in targets:
        data = pd.read_csv(f"{dir}/data/{args.dataset}/{target}.csv")
        # collect the top 80% row of the data not the first row
        # get the length of the data
        length = len(data)

        # print(length)
        if length == 1:
            now_data = process_outliers(data.iloc[0].to_list())
            normed, stats = normalize(now_data)
            test_datas.append(normed)
            task_norm_stats[target] = stats
            continue

        DATA_FRACTION = 1

        for i in range(length):
            if i % DATA_FRACTION == 0:
                tmp = process_outliers(data.iloc[i].to_list())
                normed, stats = normalize(tmp)
                train_datas.append(normed)
    dim = len(embedding[0])
    # print(task_norm_stats)
    # exit(0)
    print(f"Embedding dimension: {dim}")

    N = len(embedding)

    test_dataset = DatasetMAE(test_datas, args.seed, args.few_shot, test=1)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    return test_loader, dim, embedding, N


def test(model, test_loader, device, args, num_inference_steps=1000, indicator=None,
         stage="test", NUM_GEN_ITERS=10, NUM_RUN_ITERS=10):
    """
    在测试集上执行完整的多步生成过程，并评估最终性能。
    """

    set_random_seed(args.seed)

    model.eval()
    final_summary_metrics = {}
    with torch.no_grad():
        for labels, mask in test_loader:
            labels = labels.to(device)
            mask = mask.to(device)  # 0: train(input), 1: test, 2 eval

            if stage == "test":
                test_mask = (mask == 1)
            else:
                test_mask = (mask == 2)

            # print(stage)
            #
            # print(mask)
            #
            # print(test_mask)

            mask = (mask != 0)  # input is 0 output is 1
            mask_float = mask.float()

            known_data = labels * (1 - mask_float)  # known data is label, test data is 0
            unknown_number_in_the_beginning = 0

            for i in range(labels.shape[1]):
                if mask_float[0][i] == 1:
                    unknown_number_in_the_beginning += 1

            known_number_now = 0
            # print(known_data)
            for step in range(NUM_GEN_ITERS):
                # print(f"generating iter {step}")
                x_0s = []
                for run in range(NUM_RUN_ITERS):
                    x_t = torch.randn_like(labels)
                    x_t = known_data + x_t * mask_float  # known data is label, test data is noise
                    noise_scheduler.set_timesteps(num_inference_steps)
                    for t in noise_scheduler.timesteps:
                        t_batch = torch.full((labels.shape[0],), t, device=device, dtype=torch.long)

                        pred_noise, _, pred_mask = model(
                            x_t.unsqueeze(-1),
                            t_batch
                        )

                        # 裁剪预测值
                        x_t = noise_scheduler.step(pred_noise, t, x_t).prev_sample

                        x_t = known_data + x_t * mask_float
                    x_0s.append(x_t)

                # calculate the std of each position
                x_0s = torch.stack(x_0s, dim=1)
                x_0s_mean = x_0s.mean(dim=1)
                x_0s_std = x_0s.std(dim=1)

                for task_ in range(len(x_0s_std)):
                    indexes = [(i, x_0s_mean[task_][i], x_0s_std[task_][i]) for i in range(len(x_0s_std[0]))]
                    # sort the indexes by the std
                    indexes = sorted(indexes, key=lambda x: x[2], reverse=False)

                    ratio = 1. * (step + 1) / NUM_GEN_ITERS

                    not_selected_ratio = 1 - np.cos(math.pi / 2. * ratio) if step != NUM_GEN_ITERS - 1 else 1.0

                    not_selected_len = not_selected_ratio * unknown_number_in_the_beginning

                    temp_known_number_now = known_number_now
                    for i in range(len(indexes)):
                        if mask_float[task_][indexes[i][0]] == 1:
                            if temp_known_number_now < not_selected_len:
                                temp_known_number_now += 1
                                known_data[task_, indexes[i][0]] = x_0s_mean[task_, indexes[i][0]]
                                mask_float[task_, indexes[i][0]] = 0

                # print(known_data)

                known_number_now = temp_known_number_now

            final_prediction = known_data

            print("\n--- Test Results per Task ---")
            bool_mask = test_mask.bool()  # 创建布尔mask用于索引
            headers = ["Task", "MSE", "R2", "RMSE", "MAE", "MAPE", "PCC"]
            rows = []
            output_json = {}
            for i in range(len(tasks)):
                if indicator is not None and tasks[i] != indicator:
                    continue
                truths = labels[i][bool_mask[i]].cpu().numpy()
                output_json[tasks[i]] = {}
                # truths = denormalize(truths, task_norm_stats[tasks[i]])

                predictions = final_prediction[i][bool_mask[i]].cpu().numpy()
                # predictions = denormalize(predictions, task_norm_stats[tasks[i]])

                task_name = targets[i]
                print(f"\nResults for Task: {targets[i]}")
                task_metrics = calc("Test", targets[i], predictions, truths, None, args,
                                    model="MaskDiffusion")
                row = [task_name]
                for metric_name, metric_value in task_metrics.items():
                    summary_key = f"test/{task_name}_{metric_name}"
                    final_summary_metrics[summary_key] = metric_value
                    row.append(f"{metric_value:.4f}")
                    output_json[tasks[i]][metric_name] = metric_value
                rows.append(row)

        return final_summary_metrics, output_json


@torch.no_grad()
def update_ema(ema_model, model, decay=0.999):  # decay值通常很高
    ema_params = ema_model.state_dict()
    model_params = model.state_dict()
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param, alpha=1 - decay)


def cal_cosine_loss(zs, region_emb):
    """
    zs: [B, N, z_dim]
    region_emb: [N, z_dim]
    mask: [B, N]
    """
    B, N, D = zs.shape

    region_emb_norm = F.normalize(region_emb, dim=-1)  # [N, D]
    zs_norm = F.normalize(zs, dim=-1)  # [B, N, D]

    # 计算每个位置与对应 region_emb 的 cosine 相似度
    sim = (zs_norm * region_emb_norm.unsqueeze(0)).sum(-1)  # [B, N]

    # 目标为最大化对齐（即相似度越高越好）
    align_loss = 1.0 - sim  # [B, N]

    # 只在 unmasked 区域上计算对齐（或根据需要也可在 masked 上计算）
    # loss_align = (align_loss * (1 - mask.float())).sum() / ((1 - mask.float()).sum() + 1e-8)

    return align_loss


def train_one_epoch(model, optimizer, data_loader, epoch, num_epochs, device, global_step, scheduler, ema):
    model.train()
    # pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]")
    accumulation_steps = 2
    optimizer.zero_grad()
    for i, (labels_, mask) in enumerate(data_loader):
        labels = labels_.to(device)
        # truncted labels to -1 to 1
        # todo: delete this
        # labels = torch.clamp(labels, -1, 1)
        mask = mask.to(device)
        mask_float = mask.float()
        unmask_float = 1 - mask_float

        noise = torch.randn_like(labels)

        t_discrete = torch.randint(0, noise_scheduler.config.num_train_timesteps, (labels.shape[0],),
                                   device=device).long()
        x_t = noise_scheduler.add_noise(labels, noise, t_discrete)

        model_input = labels * unmask_float + x_t * mask_float

        # predicted_noise  = model(model_input.unsqueeze(-1), t_discrete).squeeze(-1)
        predicted_noise, aligned_embeddings, predicted_mask = model(model_input.unsqueeze(-1), t_discrete)
        predicted_noise = predicted_noise.squeeze(-1)

        target = noise
        noise_loss = F.mse_loss(predicted_noise, target, reduction='none')
        masked_noise_loss = (noise_loss * mask_float).sum() / (mask_float.sum() + 1e-8)

        # 新增: mask预测loss
        mask_target = mask_float  # 使用实际的mask作为目标
        mask_loss = F.binary_cross_entropy(predicted_mask, mask_target, reduction='none').mean()
        acc = ((predicted_mask > 0.5) == (mask_target > 0.5)).float().mean().item()
        # print(acc)

        cosine_loss = cal_cosine_loss(aligned_embeddings, model.region_emb).mean()

        # print(masked_noise_loss.item(), cosine_loss.mean().item(), mask_loss.item() )

        # 组合loss - 可以通过args.mask_loss_weight来调整mask loss的权重
        total_loss = masked_noise_loss + args.mask_loss_weight * mask_loss + 0.1 * cosine_loss

        total_loss.backward()

        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(data_loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            update_ema(ema, model)
            optimizer.zero_grad()

        if i % 20 == 0:  # 每20个step记录一次，避免过于频繁
            run.log({"train/loss": total_loss.item()}, step=global_step)

        global_step += 1
        run.log({"train/lr": scheduler.get_last_lr()[0]}, step=global_step)

    return global_step


def main(args):
    set_random_seed(args.seed)

    device = torch.device(args.gpu)

    test_loader, dim, embedding, N = create_datasets(args)

    config = {
        "d05": GPTConfig(block_size=N, input_dim=N, n_layer=1, n_head=4, n_embd=8),
        "d025": GPTConfig(block_size=N, input_dim=N, n_layer=1, n_head=4, n_embd=16),
        "d1": GPTConfig(block_size=N, input_dim=N, n_layer=1, n_head=4, n_embd=32),
        "d2": GPTConfig(block_size=N, input_dim=N, n_layer=2, n_head=8, n_embd=64),
        "d10": GPTConfig(block_size=N, input_dim=N, n_layer=4, n_head=8, n_embd=128),
        "d15": GPTConfig(block_size=N, input_dim=N, n_layer=2, n_head=4, n_embd=32),
        "d25": GPTConfig(block_size=N, input_dim=N, n_layer=4, n_head=8, n_embd=64),
        "d100": GPTConfig(block_size=N, input_dim=N, n_layer=8, n_head=16, n_embd=256),
    }[args.model]

    source = args.source
    if args.use_embedding == 0:
        embedding = None
        source = "None"
    else:
        embedding = torch.from_numpy(embedding).float().to(device)

    model = MaskFlowDiffusionGPT(config, region_emb=embedding).to(device)
   
    save_path = f"{args.checkpoint_dir}/{args.dataset}_{args.task}_best_model.pth"


    best_checkpoint = torch.load(save_path)
    model.load_state_dict(best_checkpoint)

    test_results = test(model, test_loader, device, args, num_inference_steps=args.sample_step, stage="test",
                        NUM_GEN_ITERS=1)

    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = parse_args(parser)
    args = parser.parse_args()
    settings = Settings(requirements_collect=False)

    run = swanlab.init(
        api_key='xxxxxx',
        project="xxx",  # 你的项目名称
        experiment_name=f"{args.name}-{args.model}-{args.dataset}-{args.seed}",  # 实验名称
        # 推荐：保存超参数配置
        config=vars(args),
        settings=settings
    ) if args.enable_swanlab else DummyRun()

    noise_scheduler = DDPMScheduler(num_train_timesteps=args.sample_step,
                                    beta_schedule='squaredcos_cap_v2')

    main(args)
