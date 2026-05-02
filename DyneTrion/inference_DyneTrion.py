
import os
import concurrent.futures
import torch
import GPUtil
import time
import numpy as np
import hydra
import logging
import copy
import random
import pandas as pd
import sys
from datetime import datetime
from omegaconf import DictConfig
from omegaconf import OmegaConf
from torch.utils import data
from typing import Dict

from src.data import DyneTrion_data_loader_dynamic
from src.data import utils as du
import DyneTrion.train_DyneTrion as train_DyneTrion





class Evaluator:
    def __init__(
            self,
            conf: DictConfig,
            conf_overrides:Dict=None
    ):
        self._log = logging.getLogger(__name__)

        # Remove static type checking.
        OmegaConf.set_struct(conf, False)

        # Prepare configs.
        self._conf = conf
        self._eval_conf = conf.eval
        self._diff_conf = conf.diffuser
        self._data_conf = conf.data
        self._exp_conf = conf.experiment

        # Set-up GPU
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self._conf.experiment.device = self.device
        self._log.info(f'Using device: {self.device}')
        # model weight
        self._weights_path = self._eval_conf.weights_path
        output_dir =self._eval_conf.output_dir
        self._output_dir = output_dir
        os.makedirs(self._output_dir, exist_ok=True)
        self._log.info(f'Saving results to {self._output_dir}')
        # Load models and experiment
        self._load_ckpt(conf_overrides)


        

    def _load_ckpt(self, conf_overrides):
        """Loads in model checkpoint."""
        self._log.info(f'===================>>>>>>>>>>>>>>>> Loading weights from {self._weights_path}')
        # Read checkpoint and create experiment.
        weights_pkl = du.read_pkl(self._weights_path, use_torch=True, map_location='cpu')#self.device

        # Merge base experiment config with checkpoint config.
        # self._conf.model = OmegaConf.merge(self._conf.model, weights_pkl['conf'].model)
        if conf_overrides is not None:
            self._conf = OmegaConf.merge(self._conf, conf_overrides)

        # Prepare model
        self._conf.experiment.ckpt_dir = None
        self._conf.experiment.warm_start = None
        self.exp = train_DyneTrion.Experiment(conf=self._conf)
        self.model = self.exp.model

        # Remove module prefix if it exists.
        model_weights = weights_pkl['model']
        model_weights = {k.replace('module.', ''):v for k,v in model_weights.items()}

        self.model.load_state_dict(model_weights)

        self.model = self.model.to(self.device)
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
        enable_compile = bool(getattr(self._conf.experiment, "enable_torch_compile", False))
        compile_fn = getattr(torch, "compile", None)
        if enable_compile and compile_fn is not None:
            try:
                self.model = compile_fn(self.model, dynamic=True)
                self._log.info("Enabled torch.compile(dynamic=True) for inference.")
            except Exception as exc:
                self._log.warning(f"torch.compile disabled: {exc}")
        elif enable_compile:
            self._log.warning("torch.compile is not available in this PyTorch build.")
        else:
            self._log.info("torch.compile is disabled for inference.")
        self.exp._model = self.model
        self.model.eval()
        self.diffuser = self.exp.diffuser

        self._log.info(f"Successfully loaded model weights from {self._weights_path}.")

    def create_dataset(self,is_random=False):
        test_dataset = DyneTrion_data_loader_dynamic.PdbDataset(
            data_conf=self._data_conf,
            diffuser=self.exp._diffuser,
            is_training=False,
            is_testing=True,
            is_random_test=is_random,
            rank=self._conf.eval.rank_idx,
            grouped=self._conf.eval.group,
        )
        return test_dataset

    def start_evaluation(self):
        self._log.info("Preparing evaluation dataset and warmup.")
        test_dataset = self.create_dataset(is_random=self._conf.eval.random_sample)
        num_to_run = len(test_dataset)
        self._log.info(f"Number of proteins scheduled for inference: {num_to_run}")
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

        eval_dir = self._output_dir
        os.makedirs(eval_dir, exist_ok=True)
        self._log.info(f"Eval directory: {eval_dir}")
        config_path = os.path.join(eval_dir ,'eval_conf.yaml')
        with open(config_path, 'w') as f:
            OmegaConf.save(config=self._conf, f=f)
        self._log.info(f'Saving inference config to {config_path}')
        self._log.info("Starting extrapolation evaluation.")
        self.exp._set_seed(42)
        pdb_base_path, ref_base_path = self.exp._prepare_extension_eval_dirs(eval_dir)
        extrapolation_time = self.exp._conf.eval.extrapolation_time
        len_col = "total_seq_len" if "total_seq_len" in test_dataset.csv.columns else "seq_len"
        if num_to_run > 0 and len_col in test_dataset.csv.columns:
            current_batch_df = test_dataset.csv.iloc[:num_to_run]
            max_idx_in_batch = current_batch_df[len_col].idxmax()
            max_len = current_batch_df[len_col].max()
            if "pdb_id" in current_batch_df.columns:
                pdb_id_of_max = current_batch_df.loc[max_idx_in_batch, "pdb_id"]
            else:
                pdb_id_of_max = os.path.basename(current_batch_df.loc[max_idx_in_batch, "pos_path"])
            relative_idx = current_batch_df.index.get_loc(max_idx_in_batch)
            self._log.info(
                f"Warmup will use the longest protein in this batch [ID: {pdb_id_of_max}, Length: {max_len}]."
            )
            with torch.no_grad():
                warmup_feats, _ = test_dataset._get_row(relative_idx)
                for k, v in warmup_feats.items():
                    if torch.is_tensor(v):
                        warmup_feats[k] = v.to(self.device)
                f_time, l_len = warmup_feats["res_mask"].shape
                warmup_num_t = 2
                z_rot_all = torch.randn(warmup_num_t, f_time, l_len, 3, device=self.device)
                z_trans_all = torch.randn(warmup_num_t, f_time, l_len, 3, device=self.device)
                self.exp.inference_fn(
                    warmup_feats,
                    num_t=warmup_num_t,
                    min_t=0.01,
                    aux_traj=True,
                    z_rot_all=z_rot_all,
                    z_trans_all=z_trans_all,
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                self._log.info("Warmup completed successfully; CUDA memory is initialized.")

        future = executor.submit(test_dataset._get_row, 0) if num_to_run > 0 else None
        for i in range(num_to_run):
            valid_feats, pdb_names = future.result()
            seq_len = valid_feats["aatype"].shape[-1]
            self._log.info(
                f"[Progress {i + 1}/{num_to_run}] Processing PDB: {pdb_names} | Length: {seq_len}"
            )
            if i + 1 < num_to_run:
                future = executor.submit(test_dataset._get_row, i + 1)
            for k, v in valid_feats.items():
                if torch.is_tensor(v):
                    valid_feats[k] = v.unsqueeze(0).to(self.device, non_blocking=True)
                else:
                    valid_feats[k] = v
            self.exp._process_one_protein_extrapolation(
                extrapolation_time,
                valid_feats,
                [pdb_names],
                ref_base_path,
                pdb_base_path,
                device=self.device,
                noise_scale=self.exp._exp_conf.noise_scale,
                executor=executor,
            )
        executor.shutdown(wait=True)



@hydra.main(version_base=None, config_path="./config", config_name="eval_DyneTrion")
def run(conf: DictConfig) -> None:
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    # Read model checkpoint.
    print('Starting inference')
    start_time = time.time()
    sampler = Evaluator(conf)

    sampler.start_evaluation()

    elapsed_time = time.time() - start_time
    print(f'Finished in {elapsed_time:.2f}s')

if __name__ == '__main__':
    run()
