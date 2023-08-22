import numpy as np
import cv2


class ArtConverter:
    def __init__(self, font_size=0.2, alphabet=' .:!/r(l1Z4H9W8$@'):
        self.ASCII_CHARS = alphabet
        self.font_size = font_size
        self.CHAR_STEP = int(font_size * 40)
        self.colors = {
            'green': (0, 255, 0),
            'blue': (255, 0, 0),
            'red': (0, 0, 255),
            'white': (255, 255, 255)
        }

    @staticmethod
    def get_image(path):
        image = cv2.imread(path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray_image

    def create_ascii_image_cv2(self, path: str, color: str = 'white'):
        gray_image = self.get_image(path)
        resolution = width, height = gray_image.shape[0], gray_image.shape[1]
        ascii_coeff = gray_image.max() // (len(self.ASCII_CHARS) - 1)

        black_image = np.full((*resolution, 3), 0, dtype=np.uint8)
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
    def draw_resized_cv2_image(cv2_image):
        resized_cv2_image = cv2.resize(cv2_image, (640, 360), interpolation=cv2.INTER_AREA)
        cv2.imshow('img', resized_cv2_image)

    @staticmethod
    def save_ascii_image(path, image, name='ascii'):
        cv2.imwrite(f'{path.strip(".jpg")}_{name}.jpg', image)

    def run_photo(self, path: str, color: str = 'white', name='ascii'):
        ascii_image = self.create_ascii_image_cv2(path, color)
        self.save_ascii_image(path, ascii_image, name)


if __name__ == '__main__':
    app = ArtConverter(font_size=0.05)
    app.run_photo('image/test.jpg', 'green')
