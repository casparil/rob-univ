from pathlib import Path
import torch
import pandas as pd
import itertools
import tqdm
from loguru import logger

model_dirs = [
    Path("runs", "30epochs_noNorm", "densenet161_in1k_30epochs_noNorm"),
    Path("runs", "30epochs_noNorm", "resnet18_in1k_30epochs_noNorm"),
    Path("runs", "30epochs_noNorm", "resnet50_in1k_30epochs_noNorm"),
    Path("runs", "30epochs_noNorm", "resnext_in1k_30epochs_noNorm"),
    Path("runs", "30epochs_noNorm", "vgg16_in1k_30epochs_noNorm"),
    Path("runs", "30epochs_noNorm", "wide_rn50_2_in1k_30epochs_noNorm"),
    Path("runs", "30epochs_noNorm", "wide_rn50_4_in1k_30epochs_noNorm"),
]
model_dir_to_cache_reps_name = {
    "densenet161_in1k_30epochs_noNorm": "densenet161_in1k",
    "resnet18_in1k_30epochs_noNorm": "resnet18_in1k",
    "resnet50_in1k_30epochs_noNorm": "resnet50_in1k",
    "resnext_in1k_30epochs_noNorm": "resnext50_32x4d_in1k",
    "vgg16_in1k_30epochs_noNorm": "vgg16_bn_in1k",
    "wide_rn50_2_in1k_30epochs_noNorm": "wide_resnet50_2_in1k",
    "wide_rn50_4_in1k_30epochs_noNorm": "wide_resnet50_4_in1k",
}

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
num_classes = 1000

all_preds = {}
total_probes = sum(len(list(next(model_dir.glob("*")).iterdir())) for model_dir in model_dirs)
pbar = tqdm.tqdm(total=total_probes, desc="Processing probes")

for model_dir in model_dirs:
    model_name = model_dir_to_cache_reps_name[model_dir.name]
    reps_path = Path("cache", f"reps_{model_name}_train.pt")
    logger.info(f"Loading training representations from {reps_path}")
    reps = torch.load(reps_path).to(device)

    val_reps_path = Path("cache", f"reps_{model_name}_val.pt")
    logger.info(f"Loading validation representations from {val_reps_path}")
    val_reps = torch.load(val_reps_path).to(device)

    probes_dir = next(model_dir.glob("*"))
    for probe_dir in probes_dir.iterdir():
        probe_path = probe_dir / "best_acc_checkpoint.pt"
        logger.info(f"Loading probe from {probe_path}")
        probe = torch.nn.Linear(reps.shape[1], num_classes).to(device)
        probe.load_state_dict(torch.load(probe_path))
        logger.info(f"Loaded probe from {probe_path}")

        with torch.inference_mode():
            with torch.autocast(device_type=device.type):
                logits = probe(reps).to("cpu")
                preds = logits.argmax(dim=-1)
        all_preds[probe_dir] = preds

        # Clear GPU memory
        del logits
        torch.cuda.empty_cache()

        logger.info("Completed inference")
        pbar.update(1)

    # Clear GPU memory after processing all probes for this model
    del reps, val_reps
    torch.cuda.empty_cache()

pbar.close()

agreement_results = {"probe_1": [], "probe_2": [], "agreement": []}
probe_combinations = list(itertools.combinations(all_preds.keys(), 2))
for probe_id_1, probe_id_2 in tqdm.tqdm(probe_combinations, desc="Computing probe agreements"):
    agreement = (all_preds[probe_id_1] == all_preds[probe_id_2]).float().mean().item()
    agreement_results["probe_1"].append(probe_id_1)
    agreement_results["probe_2"].append(probe_id_2)
    agreement_results["agreement"].append(agreement)

agreement_df = pd.DataFrame.from_dict(agreement_results)
agreement_df.to_csv(Path("results", "probing", "probe_agreement.csv"), index=False)
