import pygame as pg
import cv2


class ArtConverter:
    def __init__(self, path='image/egor3.jpg', font_size=12, alphabet=' .:!/r(l1Z4H9W8$@'):
        pg.init()
        self.path = path
        self.cv2_image = cv2.imread(self.path)
        self.image = self.get_image()
        self.RESOLUTION = self.WIDTH, self.HEIGHT = self.image.shape[0], self.image.shape[1]
        self.surface = pg.display.set_mode(self.RESOLUTION)

        self.ASCII_CHARS = alphabet
        self.ASCII_COEFF = self.image.max() // (len(self.ASCII_CHARS) - 1)
        self.font = pg.font.SysFont('Courier', font_size, bold=True)
        self.CHAR_STEP = int(font_size * 0.8)
        self.RENDERED_ASCII_CHARS = [self.font.render(char, False, 'white') for char in self.ASCII_CHARS]

    def draw_ascii_image(self):
        char_indices = self.image // self.ASCII_COEFF
        for x in range(0, self.WIDTH, self.CHAR_STEP):
            for y in range(0, self.HEIGHT, self.CHAR_STEP):
                char_index = char_indices[x, y]
                if char_index:
                    self.surface.blit(self.RENDERED_ASCII_CHARS[min(char_index, len(self.ASCII_CHARS) - 1)], (x, y))

    def get_image(self):
        transposed_image = cv2.transpose(self.cv2_image)
        gray_image = cv2.cvtColor(transposed_image, cv2.COLOR_BGR2GRAY)
        return gray_image

    def draw_resized_cv2_image(self):
        resized_cv2_image = cv2.resize(self.cv2_image, (640, 360), interpolation=cv2.INTER_AREA)
        cv2.imshow('img', resized_cv2_image)

    def draw(self):
        self.surface.fill('black')
        self.draw_ascii_image()

    def save_image(self):
        pygame_image = pg.surfarray.array3d(self.surface)
        cv2_img = cv2.transpose(pygame_image)
        cv2.imwrite(f'{self.path.strip(".jpg")}_converted.jpg', cv2_img)

    def run(self):
        self.draw()
        self.save_image()


if __name__ == '__main__':
    app = ArtConverter('image/sasha.jpg', alphabet=' .:!/r(l1Z4H9W8$@'[::-1])
    app.run()
