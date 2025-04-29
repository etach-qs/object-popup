import argparse
import logging
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
from popup.models.object_popup import ObjectPopup
from popup.core.generator import Generator
from popup.data.dataset import ObjectPopupDataset
from popup.utils.exp import init_experiment
import pdb
def load_input_npz(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    return data['human_verts'], str(data['obj_name'])

@torch.no_grad()
def run_popup_on_npz_files(cfg):
    device = torch.device("cuda:0")
    
    # 加载模型
    gen_datasets = []
    canonical_obj_meshes, canonical_obj_keypoints = dict(), dict()
    for dataset_name in cfg.datasets:
        if dataset_name == "grab":
            gen_datasets.append((dataset_name, ObjectPopupDataset(
                cfg, cfg.grab_path, eval_mode=True, objects=cfg.grab["gen_objects"], subjects=cfg.grab["gen_subjects"],
                actions=cfg.grab["gen_actions"]
            )))
        elif dataset_name == "behave":
            gen_datasets.append((dataset_name, ObjectPopupDataset(
                cfg, cfg.behave_path, objects=cfg.behave["gen_objects"], split_file=cfg.behave["gen_split_file"],
                eval_mode=True, downsample_factor=1
            )))
        canonical_obj_keypoints.update(gen_datasets[-1][1].canonical_obj_keypoints)
        canonical_obj_meshes.update(gen_datasets[-1][1].canonical_obj_meshes)
    import pdb; pdb.set_trace()
    model = ObjectPopup(canonical_obj_keypoints=canonical_obj_keypoints, **cfg.model_params)
    generator = Generator(device, cfg, canonical_obj_meshes, canonical_obj_keypoints)
    generator.load_checkpoint(model, cfg.checkpoint_path)
    model.eval().to(device)

    input_dir = Path(cfg.input_dir)
    all_npzs = list(input_dir.glob("*.npz"))
    output_dir = Path(cfg.exp_folder) / "visualization" / f"epoch_{cfg.epoch}"
    output_dir.mkdir(parents=True, exist_ok=True)
    pdb.set_trace()
    for npz_path in tqdm(all_npzs, desc="Processing .npz files"):
        human_verts, obj_name = load_input_npz(npz_path)
        human_verts = torch.tensor(human_verts, dtype=torch.float32).unsqueeze(0).to(device)  # (1, N, 3)

        # 获取 class id、mesh、keypoints
        obj_class_id = cfg.objname2classid[obj_name]
        obj_class_tensor = torch.tensor([obj_class_id], dtype=torch.long).to(device)

        scale = torch.ones((1, 1), dtype=torch.float32).to(device)

        output = model(human_verts, obj_class_tensor, obj_keypoints=None, obj_scales=scale, obj_center=None)

        pred_mesh, _ = generator.get_mesh_from_predictions(
            output=output[0].cpu(),
            preprocess_scale=scale[0].cpu(),
            obj_class=obj_class_tensor[0].cpu(),
            cfg=cfg,
            canonical_obj_keypoints=generator.canonical_obj_keypoints,
            canonical_obj_meshes=generator.canonical_obj_meshes,
        )

        save_path = output_dir / (npz_path.stem + ".obj")
        pred_mesh.export(str(save_path))
        logging.info(f"Saved predicted mesh to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluate and optionally generate model predictions")

    parser.add_argument("scenario", type=Path)

    parser.add_argument("-c", "--project-config", type=Path, default="./project_config.toml")
    parser.add_argument("-b", "--batch-size", type=int)
    parser.add_argument("-w", "--workers", type=int)
    parser.add_argument("--generate", "-g", action="store_true", help="Generate predictions before evaluating.")
    parser.add_argument("--downsample", action="store_true", help="Downsample datasets.")
    parser.add_argument("-d", "--datasets", type=str, nargs="+", default=None)
    parser.add_argument("--exp-name", type=str, default="experiment",
                        help="Folder name for the experiment if -rc option is used.")
    parser.add_argument("--input_path",type=str, default="/ssd1/lishujia/object-popup/for_popup")

    resume = parser.add_mutually_exclusive_group(required=True)
    resume.add_argument("-ep", "--experiment-prefix", type=str,
                        help="Prefix of the experiment to continue with the desired epoch "
                             "in format <prefix>:<epoch>. Epoch==-1 corresponds to the latest available epoch.")
    resume.add_argument("-rc", "--resume-checkpoint", type=str,
                        help="Absolute path to the checkpoint to continue.")

    arguments = parser.parse_args()
    config = init_experiment(arguments, train=False)
    run_popup_on_npz_files(config)
