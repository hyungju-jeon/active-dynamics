import os
import imageio


class VideoRecorder:
    def __init__(self, env, path):
        self.env = env
        self.frames = []
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def capture_frame(self):
        frame = self.env.render()
        self.frames.append(frame)

    def save(self):
        imageio.mimsave(self.path, self.frames, fps=20)
