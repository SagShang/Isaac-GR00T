#!/usr/bin/env python

"""Convert Franka cube-pick recordings to GR00T LeRobot v2 format with joint actions."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


DEFAULT_INPUT_DIR = Path("datasets/franka_pick_and_place_cube")
DEFAULT_OUTPUT_DIR = Path("datasets/franka_pick_and_place_cube_lerobot")
CHUNK_SIZE = 1000
VIDEO_MAP = {
    "base_rgb": "observation.images.base",
    "wrist_rgb": "observation.images.wrist",
}
FRANKA_JOINT_NAMES = [
    "fr3_joint1.pos",
    "fr3_joint2.pos",
    "fr3_joint3.pos",
    "fr3_joint4.pos",
    "fr3_joint5.pos",
    "fr3_joint6.pos",
    "fr3_joint7.pos",
]
GRIPPER_POSITION_NAME = "robotiq_85_left_knuckle_joint.pos"
GRIPPER_ACTION_NAME = "robotiq_85_gripper_action"


@dataclass(frozen=True)
class VideoInfo:
    codec_name: str
    width: int
    height: int
    pix_fmt: str
    fps: float
    frame_count: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--target-fps",
        type=float,
        default=None,
        help=(
            "Optionally downsample to this FPS. For the 60 Hz recordings, use 20 to keep "
            "every third frame."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove the output directory before conversion if it already exists.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def probe_video(path: Path) -> VideoInfo:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,pix_fmt,avg_frame_rate,nb_frames",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(result.stdout)["streams"][0]
    frame_rate = Fraction(stream["avg_frame_rate"])
    nb_frames = stream.get("nb_frames")
    return VideoInfo(
        codec_name=stream["codec_name"],
        width=int(stream["width"]),
        height=int(stream["height"]),
        pix_fmt=stream["pix_fmt"],
        fps=float(frame_rate),
        frame_count=int(nb_frames) if nb_frames and nb_frames != "N/A" else None,
    )


def validate_source_episode(episode_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metadata_path = episode_dir / "metadata.json"
    samples_path = episode_dir / "samples.jsonl"
    required_paths = [metadata_path, samples_path]
    required_paths.extend(episode_dir / f"{video_name}.mp4" for video_name in VIDEO_MAP)
    missing = [path for path in required_paths if not path.exists()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"Episode {episode_dir.name} is missing required files: {missing_text}"
        )

    metadata = json.loads(metadata_path.read_text())
    samples = read_jsonl(samples_path)
    expected_len = int(metadata["sample_count"])
    if len(samples) != expected_len:
        raise ValueError(
            f"{samples_path} has {len(samples)} rows, but metadata sample_count is {expected_len}"
        )
    return metadata, samples


def get_episode_task(metadata: dict[str, Any], metadata_path: Path) -> str:
    if "prompt" not in metadata:
        raise ValueError(f"{metadata_path} is missing required field: prompt")

    prompt = metadata["prompt"]
    if isinstance(prompt, str) and prompt.strip():
        return prompt.strip()

    raise ValueError(f"{metadata_path} field prompt must be a non-empty string")


def make_joint_position(sample: dict[str, Any]) -> np.ndarray:
    joint_position = np.asarray(sample["robot_state"]["position"], dtype=np.float32)
    if joint_position.shape != (7,):
        raise ValueError(f"Expected 7 joint positions, got shape {joint_position.shape}")
    return joint_position


def make_gripper_action(sample: dict[str, Any]) -> np.ndarray:
    if "gripper_action" not in sample:
        raise ValueError("Expected sample to contain gripper_action")
    gripper_action = np.asarray([sample["gripper_action"]], dtype=np.float32)
    if not np.isin(gripper_action, [0.0, 1.0]).all():
        raise ValueError(f"Expected binary gripper_action 0 or 1, got {gripper_action.tolist()}")
    return gripper_action


def make_gripper_position(sample: dict[str, Any]) -> np.ndarray:
    gripper_position = np.asarray(sample["gripper_position"], dtype=np.float32)
    if gripper_position.shape != (1,):
        raise ValueError(f"Expected 1 gripper position, got shape {gripper_position.shape}")
    return gripper_position


def build_episode_dataframe(
    samples: list[dict[str, Any]],
    episode_index: int,
    global_start_index: int,
    task_index: int,
) -> pd.DataFrame:
    joint_position = np.stack([make_joint_position(sample) for sample in samples], axis=0)
    gripper_position = np.stack([make_gripper_position(sample) for sample in samples], axis=0)
    gripper_action = np.stack([make_gripper_action(sample) for sample in samples], axis=0)
    states = np.concatenate([joint_position, gripper_position], axis=1).astype(np.float32)
    actions = np.concatenate([joint_position, gripper_action], axis=1).astype(np.float32)
    timestamps = np.asarray([sample["timestamp"] for sample in samples], dtype=np.float32)
    frame_indices = np.asarray([sample["frame_index"] for sample in samples], dtype=np.int64)
    length = len(samples)

    return pd.DataFrame(
        {
            "action": list(actions.astype(np.float32)),
            "observation.state": list(states.astype(np.float32)),
            "timestamp": timestamps,
            "frame_index": frame_indices,
            "episode_index": np.full(length, episode_index, dtype=np.int64),
            "index": np.arange(global_start_index, global_start_index + length, dtype=np.int64),
            "task_index": np.full(length, task_index, dtype=np.int64),
            "annotation.human.task_description": np.full(length, task_index, dtype=np.int64),
        }
    )


def video_feature(video_info: VideoInfo) -> dict[str, Any]:
    return {
        "dtype": "video",
        "shape": [video_info.height, video_info.width, 3],
        "names": ["height", "width", "channels"],
        "info": {
            "video.height": video_info.height,
            "video.width": video_info.width,
            "video.codec": video_info.codec_name,
            "video.pix_fmt": video_info.pix_fmt,
            "video.is_depth_map": False,
            "video.fps": video_info.fps,
            "video.channels": 3,
            "has_audio": False,
        },
    }


def downsample_indices(
    sample_count: int,
    source_fps: float,
    target_fps: float | None,
) -> np.ndarray:
    if target_fps is None:
        return np.arange(sample_count, dtype=np.int64)
    if target_fps <= 0:
        raise ValueError("--target-fps must be positive")
    if target_fps > source_fps:
        raise ValueError(f"--target-fps ({target_fps}) cannot exceed source FPS ({source_fps})")

    ratio = source_fps / target_fps
    rounded_ratio = round(ratio)
    if abs(ratio - rounded_ratio) < 1e-6:
        return np.arange(0, sample_count, rounded_ratio, dtype=np.int64)

    indices = np.floor(np.arange(0, sample_count, ratio)).astype(np.int64)
    return np.unique(np.clip(indices, 0, sample_count - 1))


def write_downsampled_video(
    source_path: Path,
    dest_path: Path,
    frame_indices: np.ndarray,
    target_fps: float,
) -> VideoInfo:
    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {source_path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(dest_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        target_fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Failed to open video writer: {dest_path}")

    for frame_index in frame_indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = capture.read()
        if not ok:
            writer.release()
            capture.release()
            raise RuntimeError(f"Failed to read frame {frame_index} from {source_path}")
        writer.write(frame)

    writer.release()
    capture.release()
    return probe_video(dest_path)


def build_info(
    *,
    total_episodes: int,
    total_frames: int,
    total_tasks: int,
    fps: float,
    video_infos: dict[str, VideoInfo],
) -> dict[str, Any]:
    state_names = [*FRANKA_JOINT_NAMES, GRIPPER_POSITION_NAME]
    action_names = [*FRANKA_JOINT_NAMES, GRIPPER_ACTION_NAME]
    features = {
        "action": {"dtype": "float32", "shape": [8], "names": action_names},
        "observation.state": {"dtype": "float32", "shape": [8], "names": state_names},
        "task_index": {"dtype": "int64", "shape": [1]},
        "annotation.human.task_description": {"dtype": "int64", "shape": [1]},
    }
    for source_video_key, original_key in VIDEO_MAP.items():
        features[original_key] = video_feature(video_infos[source_video_key])

    return {
        "codebase_version": "v2.1",
        "robot_type": "franka_fr3_robotiq_85_joint_20hz",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": total_tasks,
        "chunks_size": CHUNK_SIZE,
        "fps": fps,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": (
            "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
        ),
        "features": features,
    }


def build_modality() -> dict[str, Any]:
    return {
        "state": {
            "joint_position": {"start": 0, "end": 7},
            "gripper_position": {"start": 7, "end": 8},
        },
        "action": {
            "joint_position": {"start": 0, "end": 7},
            "gripper_action": {"start": 7, "end": 8},
        },
        "video": {
            "base": {"original_key": "observation.images.base"},
            "wrist": {"original_key": "observation.images.wrist"},
        },
        "annotation": {
            "human.task_description": {"original_key": "annotation.human.task_description"},
        },
    }


def prepare_output_dir(output_dir: Path, force: bool) -> None:
    if output_dir.exists():
        if not force:
            raise FileExistsError(
                f"{output_dir} already exists. Re-run with --force to replace it."
            )
        shutil.rmtree(output_dir)
    (output_dir / "meta").mkdir(parents=True)
    (output_dir / "data").mkdir()
    (output_dir / "videos").mkdir()


def convert_dataset(
    input_dir: Path,
    output_dir: Path,
    target_fps: float | None,
) -> None:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input dataset does not exist: {input_dir}")

    episode_dirs = sorted(path for path in input_dir.iterdir() if path.is_dir())
    if not episode_dirs:
        raise FileNotFoundError(f"No episode directories found under {input_dir}")

    meta_dir = output_dir / "meta"
    episode_records: list[dict[str, Any]] = []
    global_index = 0
    video_infos: dict[str, VideoInfo] | None = None
    fps_values: list[float] = []
    tasks: list[str] = []
    task_to_index: dict[str, int] = {}

    for episode_index, episode_dir in enumerate(tqdm(episode_dirs, desc="Converting episodes")):
        metadata, samples = validate_source_episode(episode_dir)
        task = get_episode_task(metadata, episode_dir / "metadata.json")
        if task not in task_to_index:
            task_to_index[task] = len(tasks)
            tasks.append(task)
        task_index = task_to_index[task]

        source_fps = float(metadata.get("sample_hz", 60.0))
        kept_indices = downsample_indices(len(samples), source_fps, target_fps)
        samples = [samples[i] for i in kept_indices]
        output_fps = target_fps or source_fps
        chunk_index = episode_index // CHUNK_SIZE
        data_chunk_dir = output_dir / "data" / f"chunk-{chunk_index:03d}"
        data_chunk_dir.mkdir(parents=True, exist_ok=True)
        video_chunk_dir = output_dir / "videos" / f"chunk-{chunk_index:03d}"

        df = build_episode_dataframe(
            samples,
            episode_index,
            global_index,
            task_index,
        )
        df.to_parquet(data_chunk_dir / f"episode_{episode_index:06d}.parquet", index=False)

        current_video_infos: dict[str, VideoInfo] = {}
        for source_video_key, original_key in VIDEO_MAP.items():
            source_video = episode_dir / f"{source_video_key}.mp4"
            source_info = probe_video(source_video)
            if source_info.frame_count is not None and source_info.frame_count != int(
                metadata["sample_count"]
            ):
                raise ValueError(
                    f"{source_video} has {source_info.frame_count} frames, "
                    f"but samples.jsonl has {metadata['sample_count']}"
                )

            dest_dir = video_chunk_dir / original_key
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_video = dest_dir / f"episode_{episode_index:06d}.mp4"
            if target_fps is None:
                shutil.copy2(source_video, dest_video)
                info = source_info
            else:
                info = write_downsampled_video(source_video, dest_video, kept_indices, output_fps)
            current_video_infos[source_video_key] = info
            if info.frame_count is not None and info.frame_count != len(samples):
                raise ValueError(
                    f"{dest_video} has {info.frame_count} frames, but parquet has {len(samples)}"
                )

        if video_infos is None:
            video_infos = current_video_infos
        else:
            for video_key, info in current_video_infos.items():
                ref = video_infos[video_key]
                if (info.width, info.height, info.fps) != (ref.width, ref.height, ref.fps):
                    raise ValueError(
                        f"Inconsistent video metadata for {episode_dir.name}/{video_key}: "
                        f"{info} vs {ref}"
                    )

        length = len(samples)
        fps_values.append(output_fps)
        episode_records.append({"episode_index": episode_index, "tasks": [task], "length": length})
        global_index += length

    assert video_infos is not None
    fps = float(np.median(fps_values))
    write_jsonl(meta_dir / "episodes.jsonl", episode_records)
    write_jsonl(
        meta_dir / "tasks.jsonl",
        [{"task_index": task_index, "task": task} for task_index, task in enumerate(tasks)],
    )

    with open(meta_dir / "info.json", "w") as f:
        json.dump(
            build_info(
                total_episodes=len(episode_records),
                total_frames=global_index,
                total_tasks=len(tasks),
                fps=fps,
                video_infos=video_infos,
            ),
            f,
            indent=4,
        )

    with open(meta_dir / "modality.json", "w") as f:
        json.dump(build_modality(), f, indent=4)


def main() -> None:
    args = parse_args()
    prepare_output_dir(args.output_dir, args.force)
    convert_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        target_fps=args.target_fps,
    )
    print(f"Dataset created at: {args.output_dir}")


if __name__ == "__main__":
    main()
