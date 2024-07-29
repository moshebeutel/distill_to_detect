import numpy as np


def gaussian_noise(x, severity=1):
    c = [0.04, 0.06, .08, .09, .10][severity - 1]

    x = np.array(x) / 255.
    return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255


def shot_noise(x, severity=1):
    c = [500, 250, 100, 75, 50][severity - 1]

    x = np.array(x) / 255.
    return np.clip(np.random.poisson(x * c) / c, 0, 1) * 255


# def impulse_noise(x, severity=1):
#     c = [.01, .02, .03, .05, .07][severity - 1]
#
#     x = sk.util.random_noise(np.array(x) / 255., mode='s&p', amount=c)
#     return np.clip(x, 0, 1) * 255
#
#
# def glass_blur(x, severity=1):
#     # sigma, max_delta, iterations
#     c = [(0.05, 1, 1), (0.25, 1, 1), (0.4, 1, 1), (0.25, 1, 2), (0.4, 1, 2)][severity - 1]
#
#     x = np.uint8(gaussian(np.array(x) / 255., sigma=c[0]) * 255)
#     # x = np.uint8(gaussian(np.array(x) / 255., sigma=c[0], multichannel=True) * 255)
#
#     # locally shuffle pixels
#     for i in range(c[2]):
#         for h in range(32 - c[1], c[1], -1):
#             for w in range(32 - c[1], c[1], -1):
#                 dx, dy = np.random.randint(-c[1], c[1], size=(2,))
#                 h_prime, w_prime = h + dy, w + dx
#                 # swap
#                 x[h, w], x[h_prime, w_prime] = x[h_prime, w_prime], x[h, w]
#
#     return np.clip(gaussian(x / 255., sigma=c[0]), 0, 1) * 255
#     # return np.clip(gaussian(x / 255., sigma=c[0], multichannel=True), 0, 1) * 255
#
# def motion_blur(x, severity=1):
#     c = [(6, 1), (6, 1.5), (6, 2), (8, 2), (9, 2.5)][severity - 1]
#
#     output = BytesIO()
#     x.save(output, format='PNG')
#     x = MotionImage(blob=output.getvalue())
#
#     x.motion_blur(radius=c[0], sigma=c[1], angle=np.random.uniform(-45, 45))
#
#     x = cv2.imdecode(np.fromstring(x.make_blob(), np.uint8),
#                      cv2.IMREAD_UNCHANGED)
#
#     if x.shape != (32, 32):
#         return np.clip(x[..., [2, 1, 0]], 0, 255)  # BGR to RGB
#     else:  # greyscale to RGB
#         return np.clip(np.array([x, x, x]).transpose((1, 2, 0)), 0, 255)

