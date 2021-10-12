import pyglet
import numpy as np

# window = pyglet.window.Window(1920, 1080, caption='My Window', resizable=True)
#
# label = pyglet.text.Label('Hello, world',
#                           font_name='Times New Roman',
#                           font_size=36,
#                           x=window.width//2, y=window.height//2,
#                           anchor_x='center', anchor_y='center')
#
#
# def on_draw():
#     window.clear()
#     label.draw()
#
#
# pyglet.app.run()


class MyWindow(pyglet.window.Window):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_draw(self):
        layer1 = np.ones((1080, 1920), dtype=int) * 255
        layer2 = np.zeros((1080, 1920), dtype=int)
        layer3 = np.zeros((1080, 1920), dtype=int)

        image = np.dstack((layer1, layer2, layer3))

        image_data = image.data.__str__()
        view = pyglet.image.ImageData(1920, 1080, 'RGB', image_data)
        view.blit(0, 0)


winObj = MyWindow(1920, 1080, 'Hello')
pyglet.app.run()