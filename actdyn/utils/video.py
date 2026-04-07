import os
import pathlib
import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt


def figure_to_rgb_array(fig) -> np.ndarray:
    """Render a matplotlib figure into an RGB uint8 frame."""
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()


def write_video_frames(
    frames: list[np.ndarray],
    path: str | pathlib.Path,
    *,
    fps: int = 60,
    codec: str = "h264",
) -> None:
    """Persist a list of RGB frames using the shared actdyn video settings."""
    if not frames:
        raise ValueError("No frames to write.")

    out_path = pathlib.Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if codec == "prores":
        iio.imwrite(
            out_path,
            np.stack(frames),
            fps=fps,
            codec="prores_ks",
            output_params=["-profile:v", "3", "-pix_fmt", "yuv422p10le"],
            plugin="ffmpeg",
        )
        return

    if codec == "h264":
        iio.imwrite(
            out_path,
            np.stack(frames),
            fps=fps,
            codec="libx264",
            output_params=[
                "-pix_fmt",
                "yuv420p",
                "-profile:v",
                "high",
                "-crf",
                "12",
                "-movflags",
                "+faststart",
            ],
        )
        return

    if codec == "lossless":
        iio.imwrite(
            out_path,
            np.stack(frames),
            fps=fps,
            codec="libx264rgb",
            output_params=["-crf", "0", "-preset", "veryslow", "-pix_fmt", "rgb24"],
            plugin="ffmpeg",
        )
        return

    iio.imwrite(out_path, np.stack(frames), fps=fps)


class VideoRecorder:
    """
    Utility class to record videos from matplotlib figures.
    """

    def __init__(self, path: pathlib.Path = pathlib.Path("video.mov"), fps=60, codec="h264"):
        self.path = path
        self.fps = fps
        self.codec = codec
        self.frames = []
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    # -------------------------------------------------------------
    def capture_frame(self, fig=None):
        if fig:
            fig.canvas.draw()
            frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            self.frames.append(frame)
            plt.close(fig)

    # -------------------------------------------------------------
    def close(self):
        """Save the recorded frames to disk in high quality."""
        if not self.frames:
            print("[WARN] No frames to save.")
            return

        print(f"[INFO] Writing {len(self.frames)} frames → {self.path}")
        write_video_frames(self.frames, self.path, fps=self.fps, codec=self.codec)
        print("[INFO] Done.")
        self.frames.clear()
