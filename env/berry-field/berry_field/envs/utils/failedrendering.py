import pyglet
from pyglet import shapes


class PygletViewer(pyglet.window.Window):

    def __init__(self, **kwargs):
        super(PygletViewer, self).__init__(**kwargs)
        self.berryColor = (255,0,0)
        self.agentColor = (255,255,255)
        self.batch = pyglet.graphics.Batch()


    def draw_berries(self, boxes, circles_in_boxes=False, radii=None):
        """ radii: required if circles_in_boxes=True draws circles at box center
        other wise draws boxes if circles_in_boxes = False
            """

        if circles_in_boxes:
            for center, radius in zip(boxes[:,:2], radii):
                x,y = center
                circle = shapes.Circle(x, y, radius, color=self.berryColor, batch=self.batch)
                circle.opacity = 250
        else:
            for x,y,width,height in boxes:
                box = shapes.Rectangle(x, y, width, height, color=self.berryColor, batch=self.batch)
                box.opacity = 250
        

    def draw_agent(self, bounding_box, iscircle=False, radius=None):
        x,y,width,height = bounding_box
        if iscircle:
            agent = shapes.Circle(x,y,radius,color=self.agentColor, batch=self.batch)
        else:
            agent = shapes.Rectangle(x,y,width,height,self.agentColor,self.batch)


    def on_draw(self):
        print("here")
        self.clear()
        print("cleared")
        self.batch.draw()
        print("out\n")


# failed render
# import pyglet
# from .rendering import PygletViewer
# def render(self):
#     if self.viewer is None:
#         self.viewer = PygletViewer(*self.observation_space_size, resizable=True)
#     bounding_box = (*self.state, *self.observation_space_size)
#     agent_bbox = (*self.state, self.agent_size, self.agent_size)
#     boxIds, boxes = self.get_Ids_and_boxes_in_view(bounding_box)
#     boxes[:,0] -= self.state[0]; boxes[:,1] -= self.state[1]
#     self.viewer.draw_berries(boxes, self.circular_berries, self.berry_radii[boxIds])
#     self.viewer.draw_agent(agent_bbox, self.circular_agent, self.agent_size/2)
#     pyglet.app.run()