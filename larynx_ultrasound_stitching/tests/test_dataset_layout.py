from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from run_dataset import discover_dataset_jobs


def test_input2_is_target_for_both_pairs(tmp_path: Path):
	dataset = tmp_path / "dataset"
	case = dataset / "case001"
	for name in ("input1", "input2", "input3"):
		(case / name).mkdir(parents=True)
		(case / name / "slice_0001.jpg").touch()
	(case / "nipple_x.txt").write_text("123\n", encoding="utf8")

	jobs = discover_dataset_jobs(dataset, tmp_path / "outputs")

	assert [job.pair for job in jobs] == ["12", "23"]
	assert all(Path(job.target).parent.name == "input2" for job in jobs)
	assert Path(jobs[0].reference).parent.name == "input1"
	assert Path(jobs[1].reference).parent.name == "input3"
	assert all(job.nipple_x_file == str(case / "nipple_x.txt") for job in jobs)


def test_only_common_slice_names_are_scheduled(tmp_path: Path):
	dataset = tmp_path / "dataset"
	case = dataset / "case001"
	for name in ("input1", "input2", "input3"):
		(case / name).mkdir(parents=True)
	(case / "input2" / "slice_0001.jpg").touch()
	(case / "input2" / "slice_0002.jpg").touch()
	(case / "input1" / "slice_0001.jpg").touch()
	(case / "input3" / "slice_0002.jpg").touch()

	jobs = discover_dataset_jobs(dataset, tmp_path / "outputs")

	assert [(job.pair, job.slice_name) for job in jobs] == [
		("12", "slice_0001.jpg"),
		("23", "slice_0002.jpg"),
	]


def test_result_summaries_separate_success_and_failure(tmp_path: Path):
	from run_dataset import write_result_summaries

	results = [
		{
			"case": "case001",
			"pair": "12",
			"slice_name": "slice_0001.jpg",
			"status": "SUCCESS",
			"output": "/output/success",
		},
		{
			"case": "case001",
			"pair": "23",
			"slice_name": "slice_0001.jpg",
			"status": "FAILED_H",
			"message": "Only 3 RANSAC inliers; required 4",
			"output": "/output/failure",
		},
	]

	write_result_summaries(results, tmp_path)

	assert (tmp_path / "successful_outputs.txt").read_text() == "/output/success"
	assert "FAILED_H" in (tmp_path / "failed_jobs.csv").read_text()
	assert '"successful": 1' in (tmp_path / "summary.json").read_text()
