import wandb
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import torch
from pathlib import Path
from numpy import array


class Logger:
    """
    Utils functions to compute and handle the statistics (saving them or send to
    wandb). It can be passed on to querier, gfn, proxy, ... to get the
    statistics of training of the generated data at real time
    """

    def __init__(
        self,
        config: dict,
        project_name: str,
        logdir: str,
        overwrite_logdir: bool,
        sampler: dict,
        run_name=None,
        tags: list = None,
        record_time: bool = False,
    ):
        self.config = config
        if run_name is None:
            date_time = datetime.today().strftime("%d/%m-%H:%M:%S")
            run_name = "{}".format(
                date_time,
            )
        self.run = wandb.init(config=config, project=project_name, name=run_name)
        self.add_tags(tags)
        self.sampler = sampler
        self.context = "0"
        self.record_time = record_time
        # Log directory
        self.logdir = Path(logdir)
        if self.logdir.exists() or overwrite_logdir:
            self.logdir.mkdir(parents=True, exist_ok=True)
        else:
            print(f"logdir {logdir} already exists! - Ending run...")
        self.test_period = (
            np.inf
            if sampler.test.period == None or sampler.test.period == -1
            else sampler.test.period
        )
        self.policy_period = (
            np.inf
            if sampler.policy.period == None or sampler.policy.period == -1
            else sampler.policy.period
        )
        self.train_period = (
            np.inf
            if sampler.train.period == None or sampler.train.period == -1
            else sampler.train.period
        )
        self.oracle_period = (
            np.inf
            if sampler.oracle.period == None or sampler.oracle.period == -1
            else sampler.oracle.period
        )

    def add_tags(self, tags: list):
        self.run.tags = self.run.tags + tags

    def set_context(self, context: int):
        self.context = str(context)

    def log_metric(self, key: str, value, step, use_context=True):
        if use_context:
            key = self.context + "/" + key
        wandb.log({key: value}, step)

    def log_histogram(self, key, value, step, use_context=True):
        if use_context:
            key = self.context + "/" + key
        fig = plt.figure()
        plt.hist(value)
        plt.title(key)
        plt.ylabel("Frequency")
        plt.xlabel(key)
        fig = wandb.Image(fig)
        wandb.log({key: fig}, step)

    def log_metrics(self, metrics: dict, step: int, use_context: bool = True):
        if use_context:
            for key, _ in metrics.items():
                key = self.context + "/" + key
        wandb.log(metrics, step)

    def log_sampler_train(
        self,
        rewards: list,
        proxy_vals: array,
        states_term: list,
        data: list,
        step: int,
        use_context: bool,
    ):
        if not step % self.train_period:
            train_metrics = dict(
                zip(
                    [
                        "mean_reward",
                        "max_reward",
                        "mean_proxy",
                        "min_proxy",
                        "max_proxy",
                        "mean_seq_length",
                        "batch_size",
                    ],
                    [
                        np.mean(rewards),
                        np.max(rewards),
                        np.mean(proxy_vals),
                        np.min(proxy_vals),
                        np.max(proxy_vals),
                        np.mean([len(state) for state in states_term]),
                        len(data),
                    ],
                )
            )
            self.log_metrics(
                train_metrics,
                use_context=use_context,
                step=step,
            )

    def log_sampler_test(
        self, corr: array, data_logq: list, step: int, use_context: bool
    ):
        if not step % self.test_period:
            test_metrics = dict(
                zip(
                    [
                        "test_corr_logq_score",
                        "test_mean_log",
                    ],
                    [
                        corr[0, 1],
                        np.mean(data_logq),
                    ],
                )
            )
            self.log_metrics(
                test_metrics,
                use_context=use_context,
                step=step,
            )

    def log_sampler_oracle(self, energies: array, step: int, use_context: bool):
        if not step % self.oracle_period:
            energies_sorted = np.sort(energies)
            dict_topk = {}
            for k in self.sampler.oracle.k:
                mean_topk = np.mean(energies_sorted[:k])
                dict_topk.update({"oracle_mean_top{}".format(k): mean_topk})
            self.log_metrics(dict_topk, use_context=use_context, step=step)

    def log_sampler_loss(
        self, losses: list, l1_error: float, kl_div, step, use_context: bool
    ):
        loss_metrics = dict(
            zip(
                [
                    "loss",
                    "term_loss",
                    "flow_loss",
                    "l1",
                    "kl",
                ],
                [loss.item() for loss in losses] + [l1_error, kl_div],
            )
        )
        self.log_metrics(
            loss_metrics,
            use_context=use_context,
            step=step,
        )

    def save_model(self, policy_path, model_path, model, step, iter=False):
        if not step % self.policy_period:
            path = policy_path.parent / Path(
                model_path.stem
                + self.context
                + "_iter{:06d}".format(step)
                + policy_path.suffix
            )
            torch.save(model.state_dict(), path)
            if iter == False:
                torch.save(model.state_dict(), policy_path)

    def log_time(self, times: dict, step: int, use_context: bool):
        if self.record_time:
            times = {"time_{}".format(k): v for k, v in times.items()}
            self.log_metrics(times, step=step, use_contxt=use_context)

    def end(self):
        wandb.finish()
