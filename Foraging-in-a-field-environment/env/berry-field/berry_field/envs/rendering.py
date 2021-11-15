import pyglet


class Viewer(pyglet.window.Window):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_minimum_size(960, 540)
        self.image = None

    def draw_image(self, image):
        self.image = image
        image_data = self.image.data.__str__()
        view = pyglet.image.ImageData(1920, 1080, 'RGB', image_data)
        view.blit(0, 0)

    def on_draw(self):
        pass
