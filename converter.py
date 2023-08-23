from numpy import full, uint8, ndarray
import cv2
import logging

logging.basicConfig(
    level=logging.INFO,
    format='#%(levelname)s: %(name)s.%(funcName)s - %(message)s'
)


class ArtConverter:
    def __init__(self, font_size=0.2, alphabet=' .:!/r(l1Z4H9W8$@'):
        self.log = logging.getLogger('ArtConverter')
        self.ASCII_CHARS = alphabet
        self.font_size = font_size
        self.CHAR_STEP = int(font_size * 40)
        self.colors = {
            'green': (0, 255, 0),
            'blue': (255, 0, 0),
            'red': (0, 0, 255),
            'purple': (255, 0, 255),
            'white': (255, 255, 255)
        }

    # Photo processing
    @staticmethod
    def get_gray_image(path: str = None, image: ndarray = None) -> ndarray:
        if path:
            image = cv2.imread(path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        return gray_image

    def create_ascii_image_cv2(self, gray_image: ndarray, color: str = 'white') -> ndarray:
        resolution = width, height = gray_image.shape[0], gray_image.shape[1]
        ascii_coeff = gray_image.max() // (len(self.ASCII_CHARS) - 1)

        black_image = full((*resolution, 3), 0, dtype=uint8)
        char_indices = gray_image // ascii_coeff
        for x in range(0, width, self.CHAR_STEP):
            for y in range(0, height, self.CHAR_STEP):
                char_index = char_indices[x, y]

                cv2.putText(black_image,
                            self.ASCII_CHARS[min(char_index, len(self.ASCII_CHARS) - 1)],
                            (y, x),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            self.font_size,
                            self.colors[color])
        return black_image

    @staticmethod
    def draw_resized_image(image: ndarray) -> None:
        resized_image = cv2.resize(image, (640, 360), interpolation=cv2.INTER_AREA)
        cv2.imshow('img', resized_image)

    def save_ascii_image(self, path: str, image: ndarray, name: str = 'ascii') -> None:
        path = f'{path.strip(".jpg")}_{name}.jpg'
        cv2.imwrite(path, image)
        self.log.info(f'Image saved to {path}')

    # Video processing
    @staticmethod
    def get_frames(path: str) -> tuple[list[ndarray], tuple[int, int], int]:
        video = cv2.VideoCapture(path)
        fps = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_size = (int(video.get(3)), int(video.get(4)))
        frames = []
        while video.isOpened():
            ret, frame = video.read()
            if ret:
                frames.append(frame)
            else:
                video.release()

        return frames, frame_size, fps

    def convert_to_ascii_frames(self, frames: list[ndarray], color: str = 'white') -> list[ndarray]:
        self.log.info('Converting to ASCII...')
        ascii_frames = []
        i = 0
        length = len(frames)
        for frame in frames:
            i += 1
            gray_frame = self.get_gray_image(image=frame)
            ascii_frames.append(self.create_ascii_image_cv2(gray_image=gray_frame, color=color))
            self.log.info(f'{i / length:.0%}')

        return ascii_frames

    def create_ascii_video(self, path: str, frames: list[ndarray], frame_size: tuple[int, int], fps: int, name='ascii'):
        path = f'{path.strip(".mp4")}_{name}.mp4'
        self.log.info(f'Saving video to {path}')
        output = cv2.VideoWriter(
            path,
            cv2.VideoWriter.fourcc(*'mp4v'),
            fps,
            frame_size
        )

        [output.write(frame) for frame in frames]

        output.release()

    # Main methods
    def run_video(self, path: str, color: str = 'white', name='ascii') -> None:
        frames, frame_size, fps = self.get_frames(path)
        ascii_frames = self.convert_to_ascii_frames(frames, color)
        self.create_ascii_video(path, ascii_frames, frame_size, fps, name)

    def run_photo(self, path: str, color: str = 'white', name='ascii') -> None:
        gray_image = self.get_gray_image(path=path)
        ascii_image = self.create_ascii_image_cv2(gray_image=gray_image, color=color)
        self.save_ascii_image(path, ascii_image, name)


if __name__ == '__main__':
    app = ArtConverter()
    app.run_photo('image/egor.jpg', color='purple', name='commit_test')
    app.run_video('image/ya2.mp4', color='purple', name='commit_test')
    cv2.destroyAllWindows()
