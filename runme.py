from connie2de import Connie2de
from connie2dl import Connie2dl
import matplotlib.pyplot as plt

model = '2dl'

# Set up model
# obj = Connieighe(Path.special('data','USC-SIPI\peppers_gray.jpg'));
# obj = Connie3dpp(Path.special('data','Kinect\1\1\0.pgm'));
if model == '2de':
    Connie2de.initialize()
    obj = Connie2de('void', 3)
if model == '2dl':
    Connie2dl.initialize()
    obj = Connie2dl('void', 3)
# obj = Connieics(Path.special('data','USC-SIPI\peppers_color.jpg'));



obj.execute().visualize('classification').visualize('clusters')
#obj.visualize('input')
plt.show()
