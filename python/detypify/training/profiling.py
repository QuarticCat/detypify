from pathlib import Path

from detypify.config import ProfilerName


def build_profiler(profiler_name: ProfilerName, run_dir: Path):
    from lightning.pytorch.profilers import AdvancedProfiler, PyTorchProfiler, SimpleProfiler

    profiler_dir = run_dir / "profiler"
    match profiler_name:
        case ProfilerName.none:
            return None
        case ProfilerName.simple:
            return SimpleProfiler(dirpath=profiler_dir, filename="simple")
        case ProfilerName.advanced:
            return AdvancedProfiler(
                dirpath=profiler_dir,
                filename="advanced",
                line_count_restriction=0.2,
            )
        case ProfilerName.pytorch:
            return PyTorchProfiler(
                dirpath=profiler_dir,
                filename="pytorch",
                sort_by_key="cuda_time_total",
            )
        case ProfilerName.trace:
            return PyTorchProfiler(
                dirpath=profiler_dir,
                filename="trace",
                export_to_chrome=True,
                record_shapes=True,
                profile_memory=True,
                sort_by_key="cuda_time_total",
            )
