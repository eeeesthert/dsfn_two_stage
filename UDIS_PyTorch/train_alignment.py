"""Train Stage 1 independently with unsupervised photometric supervision."""

import sys
from pathlib import Path as _BootstrapPath

sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parent.parent))
import argparse
import platform
from pathlib import Path
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from UDIS_PyTorch.datasets import AlignmentDataset, SyntheticHomographyDataset
from UDIS_PyTorch.losses.alignment_loss import AlignmentLoss
from UDIS_PyTorch.models.alignment import HomographyNet
from UDIS_PyTorch.utils import load_checkpoint, save_checkpoint, set_seed


def main() -> None:
	p = argparse.ArgumentParser()
	p.add_argument("--config", default="configs/alignment.yaml")
	p.add_argument("--resume")
	p.add_argument("--weights", help="Load model weights only for a new fine-tuning run.")
	p.add_argument("--dataset-root", type=Path)
	p.add_argument("--synthetic-root", type=Path)
	p.add_argument("--synthetic-perturbation", type=int, default=16)
	p.add_argument("--learning-rate", type=float, default=None)
	p.add_argument(
		"--stages",
		nargs="+",
		choices=("12", "23"),
		default=("12", "23"),
		help="ABUS pairs used for training. By default both share one alignment model.",
	)
	p.add_argument(
		"--batch-size",
		type=int,
		default=None,
		help="Override YAML batch size. Native ABUS input defaults to 1 for variable slice sizes.",
	)
	a = p.parse_args()
	if a.resume and a.weights:
		p.error("--resume and --weights are mutually exclusive")
	if a.dataset_root and a.synthetic_root:
		p.error("--dataset-root and --synthetic-root are mutually exclusive")
	with open(a.config, encoding="utf8") as stream:
		cfg = yaml.safe_load(stream)
	set_seed(cfg["seed"])
	dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(
		platform.python_version(),
		torch.__version__,
		torch.version.cuda,
		torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
		cfg,
	)
	dataset_options = {
		"brightness": (cfg["brightness_min"], cfg["brightness_max"]),
		"color": (cfg["color_min"], cfg["color_max"]),
	}
	if a.synthetic_root:
		ds = SyntheticHomographyDataset(
			a.synthetic_root,
			patch_size=cfg["input_size"],
			perturbation=a.synthetic_perturbation,
		)
	elif a.dataset_root:
		dataset_options.update(abus_root=a.dataset_root, stages=tuple(a.stages))
		ds = AlignmentDataset(**dataset_options)
	else:
		dataset_options["manifest"] = cfg["manifest"]
		ds = AlignmentDataset(**dataset_options)
	batch_size = a.batch_size or (1 if a.dataset_root else cfg["batch_size"])
	dl = DataLoader(ds, batch_size, shuffle=True, drop_last=False)
	model = HomographyNet(cfg["dropout"], tuple(cfg["search_ranges"].values())).to(dev)
	lossfn = AlignmentLoss(tuple(cfg["loss_weights"].values())).to(dev)
	learning_rate = a.learning_rate or cfg["learning_rate"]
	opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
	sch = torch.optim.lr_scheduler.LambdaLR(
		opt, lambda s: cfg["lr_decay_rate"] ** (s / cfg["lr_decay_steps"])
	)
	step = 0
	if a.weights:
		state = torch.load(a.weights, map_location=dev)
		model.load_state_dict(state["model"])
	if a.resume:
		step = load_checkpoint(a.resume, model, opt, sch, dev)["step"]
		if a.learning_rate is not None:
			for group in opt.param_groups:
				group["lr"] = a.learning_rate
	out = Path(cfg["checkpoint_dir"])
	out.mkdir(parents=True, exist_ok=True)
	writer = SummaryWriter(out / "alignment_logs")
	it = iter(dl)
	while step < cfg["iterations"]:
		try:
			b = next(it)
		except StopIteration:
			it = iter(dl)
			b = next(it)
		original1, original2 = b["image1"].to(dev), b["image2"].to(dev)
		pred = model(b["aug1"].to(dev), b["aug2"].to(dev))
		losses = lossfn(original1, original2, pred)
		if not torch.isfinite(losses["total"]):
			raise RuntimeError(f"non-finite alignment loss at step {step}")
		opt.zero_grad()
		losses["total"].backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip_norm"])
		opt.step()
		sch.step()
		step += 1
		writer.add_scalar("total_loss", losses["total"], step)
		writer.add_scalar("loss/level1", losses["level1"], step)
		writer.add_scalar("loss/level2", losses["level2"], step)
		writer.add_scalar("loss/level3", losses["level3"], step)
		writer.add_scalar("offset/mean_abs", pred["delta_final"].abs().mean(), step)
		writer.add_scalar("offset/max_abs", pred["delta_final"].abs().max(), step)
		writer.add_scalar("learning_rate", sch.get_last_lr()[0], step)
		if step % 1000 == 0:
			save_checkpoint(out / "alignment_latest.pth", model, opt, step, cfg, sch)


if __name__ == "__main__":
	main()
