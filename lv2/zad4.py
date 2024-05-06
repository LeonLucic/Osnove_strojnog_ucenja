import numpy as np
import matplotlib.pyplot as plt

black_square = np.zeros((50,50))
white_square = np.ones((50,50))
white_square *= 255

black_white = np.hstack((black_square, white_square))
white_black = np.hstack((white_square, black_square))

final_square = np.vstack((black_white, white_black))
plt.imshow(final_square, cmap="gray")
plt.show()