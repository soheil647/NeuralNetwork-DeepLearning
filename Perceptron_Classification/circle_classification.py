import numpy as np
import pylab as plt
import math

number_of_circles = np.random.randint(2, 8)
circle_radius = np.random.randint(1, 2, number_of_circles)
centers = np.random.randint(2, 5, (number_of_circles, 2))
print(number_of_circles)
print(circle_radius)
print(centers)


def generate_circle_dots(center, radius):
    dots_x = []
    dots_y = []
    for i in range(0, radius*10):
        print(np.power(np.power(radius, 2) - np.power(center[0] - (i / 10), 2), 0.5) + center[1])
        dots_x.append(center[0] - (i / 10))
        dots_y.append(math.sqrt(np.power(radius, 2) - np.power(center[0] - (i / 10), 2)) + center[1])
    return dots_x, dots_y

def generate_dots(center, radius):
    for i in range(0, radius*10):



# for circle_number in range(number_of_circles):
x, y = generate_circle_dots([2, 2], 3)
plt.plot(x, y, 'ro')
plt.show()
# circle = []
# for i in range(number_of_circles):
#     circle.append(plt.Circle((centers[i][0], centers[i][1]), radius=circle_radius[i]))
#
# ax = plt.gca()
# ax.set_xlim((0, 10))
# ax.set_ylim((0, 10))
# for i in range(len(circle)):
#     ax.add_artist(circle[i])
# plt.show()

# print(np.random.normal(-5, 5, (1, 2)))
# for i in range(number_of_circles):
#     centers.append(0.5 * np.random.normal(-5, 5, (1, 2)))
# print(centers)
# print(centers[1][0], centers[1][1])
# x = []
# for i in range(number_of_circles - 1):
#     while True:
#         x.append(0.5 * np.random.normal(-5, 5, (1, 2)))
#         print(x[1][0], x[1][1])
#         counter = 0
#         for j in range(len(centers)):
#             dist = circle_radius[i] + circle_radius[i]
#             if dist > distance(centers[j][0], centers[j][1], x[0][0], x[0][1]):
#                 counter = 1
#         if counter == 0:
#             centers = np.block([[centers], [x]])
#             break
