from PIL import Image
import numpy as np

def color_diff(c1, c2):
    return (sum((int(a) - int(b)) ** 2 for a, b in zip(c1, c2))) ** 0.5

def rgn_grwing(seed, t, img_np):
    height, width = img_np.shape[0], img_np.shape[1]
    visited = np.zeros((height, width), dtype=bool)
    rgn = np.zeros((height, width), dtype=np.uint8)
    stack = [seed]
    seed_color = img_np[seed[0]][seed[1]]

    while stack:
        x, y = stack.pop()
        if visited[x, y]:
            continue
        visited[x, y] = True
        curr_color = img_np[x, y]

        if color_diff(seed_color, curr_color) < t:
            rgn[x, y] = 255  # Use 255 directly instead of later multiplication

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < height and 0 <= ny < width and not visited[nx, ny]:
                    stack.append((nx, ny))

    return Image.fromarray(rgn)