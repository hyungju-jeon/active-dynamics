import os
import pathlib
import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt


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

        if self.codec == "prores":
            # Apple Keynote-safe, visually lossless
            iio.imwrite(
                self.path,
                np.stack(self.frames),
                fps=self.fps,
                codec="prores_ks",
                output_params=["-profile:v", "3", "-pix_fmt", "yuv422p10le"],
                plugin="ffmpeg",
            )
        elif self.codec == "h264":
            # Smaller, good quality H.264 baseline
            iio.imwrite(
                self.path,
                np.stack(self.frames),
                fps=self.fps,
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
        elif self.codec == "lossless":
            # True lossless RGB (not Keynote compatible)
            iio.imwrite(
                self.path,
                np.stack(self.frames),
                fps=self.fps,
                codec="libx264rgb",
                output_params=["-crf", "0", "-preset", "veryslow", "-pix_fmt", "rgb24"],
                plugin="ffmpeg",
            )
        else:
            # fallback: simple imageio default (quick, generic)
            iio.imwrite(self.path, np.stack(self.frames), fps=self.fps)

        print("[INFO] Done.")
        self.frames.clear()
