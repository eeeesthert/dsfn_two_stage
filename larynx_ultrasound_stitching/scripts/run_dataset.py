"""Run the supported ABUS-derived two-dimensional dataset layout."""

import argparse
import json
import logging
import sys
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

LOG = logging.getLogger(__name__)
SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class PairJob:
	"""One target-to-reference stitching job."""

	case: str
	pair: str
	slice_name: str
	reference: str
	target: str
	output: str
	nipple_x_file: str | None


def image_files(directory: Path) -> dict[str, Path]:
	"""Index supported images by filename in one input directory."""

	if not directory.is_dir():
		raise FileNotFoundError(f"Missing input directory: {directory}")
	return {
		path.name: path
		for path in sorted(directory.iterdir())
		if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
	}


def discover_case_jobs(case_dir: Path, output_root: Path) -> list[PairJob]:
	"""Create 1-to-2 and 3-to-2 jobs with input2 fixed as target.

	Pair ``12`` uses ``input1`` as reference and ``input2`` as target.
	Pair ``23`` uses ``input3`` as reference and ``input2`` as target.
	Only filenames present in target and the respective reference are scheduled.
	"""

	inputs = {
		name: image_files(case_dir / name)
		for name in ("input1", "input2", "input3")
	}
	target_files = inputs["input2"]
	nipple_path = case_dir / "nipple_x.txt"
	nipple_value = str(nipple_path) if nipple_path.is_file() else None
	jobs = []
	for pair_name, reference_name in (("12", "input1"), ("23", "input3")):
		common_names = sorted(target_files.keys() & inputs[reference_name].keys())
		missing = sorted(target_files.keys() - inputs[reference_name].keys())
		for filename in missing:
			LOG.warning(
				"Skipping %s/%s/%s because the reference slice is missing",
				case_dir.name,
				pair_name,
				filename,
			)
		for filename in common_names:
			jobs.append(
				PairJob(
					case=case_dir.name,
					pair=pair_name,
					slice_name=filename,
					reference=str(inputs[reference_name][filename]),
					target=str(target_files[filename]),
					output=str(output_root / case_dir.name / f"pair{pair_name}" / Path(filename).stem),
					nipple_x_file=nipple_value,
				)
			)
	return jobs


def discover_dataset_jobs(dataset_root: Path, output_root: Path) -> list[PairJob]:
	"""Discover all ``case*`` directories in deterministic name order."""

	if not dataset_root.is_dir():
		raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
	jobs = []
	for case_dir in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
		try:
			jobs.extend(discover_case_jobs(case_dir, output_root))
		except FileNotFoundError as error:
			LOG.warning("Skipping %s: %s", case_dir.name, error)
	return jobs


def run_jobs(jobs: list[PairJob], config_path: Path, mode: str) -> list[dict]:
	"""Run jobs independently so one failed slice does not stop the dataset."""

	from src.image import load_config
	from src.pipeline import run_pipeline

	config = load_config(config_path)
	results = []
	for index, job in enumerate(jobs, start=1):
		LOG.info(
			"[%d/%d] %s pair%s %s: reference=%s target=%s",
			index,
			len(jobs),
			job.case,
			job.pair,
			job.slice_name,
			job.reference,
			job.target,
		)
		try:
			status = run_pipeline(
				job.reference,
				job.target,
				job.output,
				config,
				mode,
			)
		except Exception as error:
			LOG.exception("Unhandled failure for %s", job)
			status = {
				"status": "FAILED_H",
				"message": str(error),
			}
		results.append({**asdict(job), **status})
	return results


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset-root", required=True, type=Path)
	parser.add_argument("--output", required=True, type=Path)
	parser.add_argument(
		"--config",
		type=Path,
		default=PROJECT_ROOT / "config.yaml",
	)
	parser.add_argument(
		"--mode",
		choices=(
			"sift",
			"sift_slic",
			"sift_slic_gabor",
			"sift_slic_saliency",
			"full",
		),
		default="full",
	)
	parser.add_argument(
		"--dry-run",
		action="store_true",
		help="Discover pairs and write the manifest without importing image-processing dependencies.",
	)
	return parser.parse_args()


def main() -> int:
	args = parse_args()
	logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
	jobs = discover_dataset_jobs(args.dataset_root, args.output)
	args.output.mkdir(parents=True, exist_ok=True)
	manifest_path = args.output / "jobs.json"
	manifest_path.write_text(
		json.dumps([asdict(job) for job in jobs], indent=2),
		encoding="utf8",
	)
	LOG.info("Discovered %d jobs. Manifest: %s", len(jobs), manifest_path)
	if args.dry_run:
		return 0
	results = run_jobs(jobs, args.config, args.mode)
	status_path = args.output / "batch_status.json"
	status_path.write_text(json.dumps(results, indent=2), encoding="utf8")
	failed = sum(result.get("status") != "SUCCESS" for result in results)
	LOG.info("Completed %d jobs with %d failures", len(results), failed)
	return 0 if failed == 0 else 2


if __name__ == "__main__":
	raise SystemExit(main())
