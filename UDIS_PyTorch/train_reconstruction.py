"""Train Stage 2 from pre-generated aligned data (never runs Stage 1)."""

import sys
from pathlib import Path as _BootstrapPath

sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parent.parent))
import argparse, platform
from pathlib import Path
import torch, yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from UDIS_PyTorch.datasets import ReconstructionDataset
from UDIS_PyTorch.losses.reconstruction_loss import ReconstructionLoss
from UDIS_PyTorch.models.reconstruction_net import ReconstructionNet
from UDIS_PyTorch.models.vgg_perceptual import VGGPerceptualExtractor
from UDIS_PyTorch.utils.utilities import load_checkpoint, save_checkpoint
from UDIS_PyTorch.utils.utilities import set_seed


def main():
	p = argparse.ArgumentParser()
	p.add_argument("--config", default="configs/reconstruction.yaml")
	p.add_argument("--resume")
	a = p.parse_args()
	cfg = yaml.safe_load(open(a.config))
	set_seed(cfg["seed"])
	dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(
		platform.python_version(),
		torch.__version__,
		torch.version.cuda,
		torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
		cfg,
	)
	dl = DataLoader(
		ReconstructionDataset(cfg["dataset"], cfg["max_image_size"]),
		cfg["batch_size"],
		shuffle=True,
	)
	model = ReconstructionNet(cfg["lr_size"], cfg["num_res_blocks"]).to(dev)
	vgg = VGGPerceptualExtractor().to(dev)
	criterion = ReconstructionLoss(
		vgg,
		cfg["seam_weight"],
		cfg["content_weight"],
		cfg["lr_loss_weight"],
		cfg["hr_loss_weight"],
		cfg["consistency_weight"],
		cfg["lr_size"],
		cfg["vgg_size"],
	)
	opt = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
	sch = torch.optim.lr_scheduler.LambdaLR(
		opt, lambda s: cfg["lr_decay_rate"] ** (s / cfg["lr_decay_steps"])
	)
	step = 0
	if a.resume:
		step = load_checkpoint(a.resume, model, opt, sch, dev)["step"]
	out = Path(cfg["checkpoint_dir"])
	out.mkdir(exist_ok=True, parents=True)
	writer = SummaryWriter(out / "reconstruction_logs")
	it = iter(dl)
	while step < cfg["iterations"]:
		try:
			b = next(it)
		except StopIteration:
			it = iter(dl)
			b = next(it)
		tensors = [b[k].to(dev) for k in ("warp1", "warp2", "mask1", "mask2")]
		pred = model(*tensors[:2])
		losses = criterion(pred, *tensors)
		opt.zero_grad()
		losses["total_loss"].backward()
		opt.step()
		sch.step()
		step += 1
		for k, v in losses.items():
			if k.endswith("loss"):
				writer.add_scalar(k, v, step)
		if step % 1000 == 0:
			save_checkpoint(out / "reconstruction_latest.pth", model, opt, step, cfg, sch)


if __name__ == "__main__":
	main()
