"""
Module takes a screenshot of a hexcells level and parses it into
the needed representation
"""
import time

import numpy as np
import pyautogui as pag
import os
from PIL import Image

from representation import Level, Cell, CellType, Line, LineDirection, ConnectivityType
from number_classifier import classify_digit, IMAGE_SIZE, TRAIN_DIRECTORY

SAVE_IMAGES = False
LOGO_DIRECTORY = "hexcells_logo"
ORANGE = 185
GRAY = 62
BLUE = 125
NO_SHAPE_THRESHOLD = 4
WHITE_THRESHOLD = 233
BACKGROUND_THRESHOLD = 60
HORIZONTAL_DISTANCE = 55
VERTICAL_HALF_DISTANCE = 31
colours = [ORANGE, BLUE, GRAY]
X_START = 0
X_END = 1920 - 1
Y_START = 0
Y_END = 1080 - 1
SEARCH_STEP = 20
X_SAFEZONE = 0.88
Y_DEADZONE = 0.19
UPPER_LEFT_DANGERZONE_MODIFIER = 0.156
UPPER_LEFT_DANGERZONE = X_END * UPPER_LEFT_DANGERZONE_MODIFIER
X_SAFE_LIMIT = int(X_END * X_SAFEZONE)
Y_SAFE_START = int(Y_END * Y_DEADZONE)
THIRTY_DEGREE_UNIT_VECTOR = (0.8944271909999159, 0.44721359549995804)


def find_extremity(screenshot: np.ndarray, start: int, limit: int, step: int):
    res = start
    while start != limit and screenshot[res + step] > BACKGROUND_THRESHOLD:
        res += step
    return res


def center(screenshot: np.ndarray, x: int, y: int):
    x_bot = find_extremity(screenshot=screenshot[y, :], start=x, limit=X_START, step=-1)
    x_top = find_extremity(screenshot=screenshot[y, :], start=x, limit=X_END, step=1)
    x_center = int((x_bot + x_top) / 2)
    y_bot = find_extremity(screenshot=screenshot[:, x], start=y, limit=Y_START, step=-1)
    y_top = find_extremity(screenshot=screenshot[:, x], start=y, limit=Y_END, step=1)
    y_center = int((y_bot + y_top) / 2)
    return x_center, y_center


def find_left_edge(screenshot: np.ndarray):
    for x in range(X_START, X_END, SEARCH_STEP):
        for y in range(Y_START, Y_END, SEARCH_STEP):
            if x + y > UPPER_LEFT_DANGERZONE \
                    and colours.__contains__(screenshot[y, x]):
                return center(screenshot=screenshot, x=x, y=y)


def find_right_edge(screenshot: np.ndarray):
    for x in range(X_END, X_START, -SEARCH_STEP):
        for y in range(Y_START, Y_END, SEARCH_STEP):
            if (x < X_SAFE_LIMIT or y > Y_SAFE_START) \
                    and colours.__contains__(screenshot[y, x]):
                return center(screenshot=screenshot, x=x, y=y)


def find_top_edge(screenshot: np.ndarray):
    for y in range(Y_START, Y_END, SEARCH_STEP):
        for x in range(X_START, X_END, SEARCH_STEP):
            if x + y > UPPER_LEFT_DANGERZONE and \
                    (y > Y_SAFE_START or x < X_SAFE_LIMIT) \
                    and colours.__contains__(screenshot[y, x]):
                return center(screenshot=screenshot, x=x, y=y)


def find_bot_edge(screenshot: np.array):
    for y in range(Y_END, Y_START, -SEARCH_STEP):
        for x in range(X_START, X_END, SEARCH_STEP):
            if colours.__contains__(screenshot[y, x]):
                return center(screenshot=screenshot, x=x, y=y)


def find_shape_measures(shape: set[(int, int)]):
    x_min = X_END
    x_max = X_START
    y_min = Y_END
    y_max = Y_START
    for (x, y) in shape:
        if x < x_min:
            x_min = x
        if x > x_max:
            x_max = x
        if y < y_min:
            y_min = y
        if y > y_max:
            y_max = y
    return x_min, y_min, x_max - x_min + 1, y_max - y_min + 1


def image_from_shape(shape: set[(int, int)], rotation_correction: int = 0):
    x_start, y_start, x_size, y_size = find_shape_measures(shape=shape)
    digit = np.zeros(shape=(y_size, x_size))
    for (x, y) in shape:
        digit[y - y_start, x - x_start] = 255

    image = Image.fromarray(np.uint8(digit), mode='L')
    new_size = max(image.size)
    y_offset = int((new_size - image.size[0]) / 2)
    x_offset = int((new_size - image.size[1]) / 2)
    padded_image = Image.new(mode=image.mode, size=(new_size, new_size), color=0)
    padded_image.paste(im=image, box=(y_offset, x_offset))
    return padded_image.rotate(angle=rotation_correction)


def identify_connectivity_type(shape: set[(int, int)], rotation_correction: int = 0):
    connectivity_indicator = np.array(image_from_shape(shape=shape, rotation_correction=rotation_correction))

    left = connectivity_indicator.shape[1]
    right = 0
    top = connectivity_indicator.shape[0]
    bot = 0
    for y in range(connectivity_indicator.shape[0]):
        for x in range(connectivity_indicator.shape[1]):
            if connectivity_indicator[y, x] > 100:  # grayscale 0-255, some random threshold
                left = min(left, x)
                right = max(right, x)
                top = min(top, y)
                bot = max(bot, y)

    if right - left >= bot - top:
        return ConnectivityType.DISJOINTED
    else:
        return ConnectivityType.CONNECTED


def identify_digit(shape: set[(int, int)], rotation_correction: int = 0):
    image = image_from_shape(shape=shape, rotation_correction=rotation_correction)
    image = image.resize(size=(IMAGE_SIZE[1], IMAGE_SIZE[0]))

    if SAVE_IMAGES:
        image.save(f"{TRAIN_DIRECTORY}/{identify_digit.image_id}.png")
        identify_digit.image_id += 1

    return classify_digit(np.array(image))


if SAVE_IMAGES:
    identify_digit.image_id = 0
    for image_name in os.listdir(TRAIN_DIRECTORY):
        image_id = int(image_name.removesuffix(".png"))
        if image_id > identify_digit.image_id:
            identify_digit.image_id = image_id
    identify_digit.image_id += 1


def in_bounds(x, y):
    return X_START <= x < X_END and \
           Y_START <= y < Y_END and \
           (x < X_SAFE_LIMIT or y > Y_SAFE_START) and \
           x + y > UPPER_LEFT_DANGERZONE


def measure_shape(screenshot: np.ndarray, x: int, y: int, shape: set[(int, int)]):
    if X_START <= x < X_END and Y_START <= y < Y_END and screenshot[y, x] >= WHITE_THRESHOLD and (x, y) not in shape:
        shape.add((x, y))
        measure_shape(screenshot=screenshot, x=x + 1, y=y, shape=shape)
        measure_shape(screenshot=screenshot, x=x - 1, y=y, shape=shape)
        measure_shape(screenshot=screenshot, x=x, y=y + 1, shape=shape)
        measure_shape(screenshot=screenshot, x=x, y=y - 1, shape=shape)
    return shape


def check_pixel_for_shape(screenshot: np.ndarray, x: int, y: int, shapes: list[set[(int, int)]]):
    if X_START <= x < X_END and Y_START <= y < Y_END and screenshot[y, x] >= WHITE_THRESHOLD:
        known = False
        for shape in shapes:
            if (x, y) in shape:
                known = True
                break
        if not known:
            shape = set()
            measure_shape(screenshot=screenshot, x=x, y=y, shape=shape)
            shapes.append(shape)


def find_number_information(shapes: list[set[(int, int)]], rotation_correction: int = 0):
    # Sometimes small groups of pixels make their way into the list of shapes
    for shape in shapes:
        if len(shape) < NO_SHAPE_THRESHOLD:
            shapes.remove(shape)

    number = None
    connectivity_type = ConnectivityType.UNKNOWN
    if len(shapes) == 1:  # one digit
        number = identify_digit(shape=shapes[0], rotation_correction=rotation_correction)
    elif len(shapes) == 2:  # two digits
        number = 10 * identify_digit(shape=shapes[0], rotation_correction=rotation_correction) \
                 + identify_digit(shape=shapes[1], rotation_correction=rotation_correction)
    elif len(shapes) == 3:  # one digit and two connectivity markers
        number = identify_digit(shape=shapes[1], rotation_correction=rotation_correction)
        connectivity_type = identify_connectivity_type(shape=shapes[0], rotation_correction=rotation_correction)
    elif len(shapes) == 4:  # two digits and two connectivity markers
        number = 10 * identify_digit(shape=shapes[1], rotation_correction=rotation_correction) \
                 + identify_digit(shape=shapes[2], rotation_correction=rotation_correction)
        connectivity_type = identify_connectivity_type(shape=shapes[0], rotation_correction=rotation_correction)
    return number, connectivity_type


def find_cell_number(screenshot: np.ndarray, x: int, y: int, cell_colour: int):
    shapes = list()
    x_temp = x
    while screenshot[y, x_temp] >= cell_colour:
        x_temp -= 1
    x_temp += 1
    while screenshot[y, x_temp] >= cell_colour:
        # The two lines signaling that blue cells are disjointed are slightly offset downwards
        # TODO: find more elegant solution
        for y_temp in range(y, y + 5):
            check_pixel_for_shape(screenshot=screenshot, x=x_temp, y=y_temp, shapes=shapes)
        x_temp += 1
    return find_number_information(shapes=shapes)


def identify_cell(screenshot: np.ndarray, x: int, y: int):
    if screenshot[y, x] <= BACKGROUND_THRESHOLD:
        return None

    y_temp = y
    # Sometimes a number contains a pixel with the same grayscale value as number cells
    # TODO: find proper solution
    while screenshot[y_temp, x] not in colours or screenshot[y_temp + 1, x] not in colours \
            or screenshot[y_temp, x] != screenshot[y_temp + 1, x]:
        y_temp += 2
    colour = screenshot[y_temp, x]
    cell_type = CellType(colours.index(colour))

    if cell_type == CellType.ORANGE:
        return Cell(cell_type=cell_type, connectivity_type=ConnectivityType.UNKNOWN)

    number, connectivity_type = find_cell_number(screenshot=screenshot, x=x, y=y, cell_colour=colour)
    return Cell(cell_type=cell_type, connectivity_type=connectivity_type, number=number)


def init_cells(screenshot: np.ndarray, level: Level):
    cells = np.zeros((level.rows, level.cols), Cell)
    orange_cells = 0
    x = level.left
    for col in range(level.cols):
        y = level.top
        if col % 2 == 1:
            y += level.vertical_half_distance
        for row in range(level.rows):
            if (x < X_SAFE_LIMIT or y > Y_SAFE_START) and x + y > UPPER_LEFT_DANGERZONE:
                cells[row, col] = identify_cell(screenshot=screenshot, x=x, y=y)
                if cells[row, col] is not None and cells[row, col].cell_type == CellType.ORANGE:
                    orange_cells += 1
            else:
                cells[row, col] = None
            y += level.vertical_half_distance * 2
        x += level.horizontal_distance
    return cells, orange_cells


def find_direction(x: int, y: int, shapes: list[set[(int, int)]], potential_directions: list[LineDirection]):
    direction_pixel_count = [0, 0, 0, 0, 0, 0]
    direction = potential_directions[0]
    most_pixels = 0
    for shape in shapes:
        for (x_pixel, y_pixel) in shape:
            x_diff = x_pixel - x
            y_diff = y - y_pixel
            direction_code = 0
            if x_diff * -THIRTY_DEGREE_UNIT_VECTOR[0] + y_diff * THIRTY_DEGREE_UNIT_VECTOR[1] > 0:
                direction_code = direction_code | 4
            if y_diff > 0:
                direction_code = direction_code | 2
            if x_diff * THIRTY_DEGREE_UNIT_VECTOR[0] + y_diff * THIRTY_DEGREE_UNIT_VECTOR[1] > 0:
                direction_code = direction_code | 1
            pixel_direction = [LineDirection.DOWN, LineDirection.DOWN_RIGHT, None, LineDirection.UP_RIGHT,
                               LineDirection.DOWN_LEFT, None, LineDirection.UP_LEFT, LineDirection.UP][direction_code]
            if pixel_direction in potential_directions:
                direction_pixel_count[pixel_direction.value] += 1
                if direction_pixel_count[pixel_direction.value] > most_pixels:
                    most_pixels = direction_pixel_count[pixel_direction.value]
                    direction = pixel_direction
    return direction


def check_line(screenshot: np.ndarray, level: Level, x: int, y: int, row: int, col: int,
               potential_directions: list[LineDirection]):
    search_range = int(level.vertical_half_distance * 0.7)
    shapes = list()
    for y_search in range(y - search_range, y + search_range):
        for x_search in range(x - search_range, x + search_range):
            check_pixel_for_shape(screenshot=screenshot, x=x_search, y=y_search, shapes=shapes)

    if len(shapes) == 0:
        return None

    line_direction = find_direction(x=x, y=y, shapes=shapes, potential_directions=potential_directions)

    if col % 2 == 0:
        start_row = row + [0, 1, 0, -1, -1, -1][line_direction.value]
    else:
        start_row = row + [1, 1, 1, 0, -1, 0][line_direction.value]
    start_col = col + [1, 0, -1, -1, 0, 1][line_direction.value]

    if len(shapes) > 1:
        sort_key = [lambda shape: -max(map(lambda pixel: pixel[1], shape)),
                    lambda shape: max(map(lambda pixel: pixel[0], shape)),
                    lambda shape: max(map(lambda pixel: pixel[1], shape)),
                    lambda shape: max(map(lambda pixel: pixel[1], shape)),
                    lambda shape: -max(map(lambda pixel: pixel[0], shape)),
                    lambda shape: -max(map(lambda pixel: pixel[1], shape))][line_direction.value]
        shapes.sort(key=sort_key)
    rotation_correction = [-60, 0, 60, 120, 180, -120][line_direction.value]
    number, connectivity_type = find_number_information(shapes=shapes, rotation_correction=rotation_correction)

    return Line(start_row=start_row, start_col=start_col, number=number, connectivity_type=connectivity_type,
                direction=line_direction)


def init_lines(screenshot: np.ndarray, level: Level):
    lines = list()
    for row in range(-1, level.rows + 1):
        for col in range(-1, level.cols + 1):
            if row == -1 or row == level.rows or col == -1 or col == level.cols or level.cells[row, col] is None:
                x, y = level.cell_coordinates(row=row, col=col)
                if not in_bounds(x, y):
                    continue
                potential_directions = list()
                if row < level.rows - 1 and 0 <= col < level.cols and level.cells[row + 1, col] is not None:
                    potential_directions.append(LineDirection.DOWN)
                if col > 0:
                    if col % 2 == 0:
                        if row < level.rows and level.cells[row, col - 1] is not None:
                            potential_directions.append(LineDirection.DOWN_LEFT)
                        if row > 0 and level.cells[row - 1, col - 1] is not None:
                            potential_directions.append(LineDirection.UP_LEFT)
                    else:
                        if row < level.rows - 1 and level.cells[row + 1, col - 1] is not None:
                            potential_directions.append(LineDirection.DOWN_LEFT)
                        if row < level.rows and level.cells[row, col - 1] is not None:
                            potential_directions.append(LineDirection.UP_LEFT)
                if row > 0 and 0 <= col < level.cols and level.cells[row - 1, col] is not None:
                    potential_directions.append(LineDirection.UP)
                if col < level.cols - 1:
                    if col % 2 == 0:
                        if row > 0 and level.cells[row - 1, col + 1] is not None:
                            potential_directions.append(LineDirection.UP_RIGHT)
                        if row < level.rows and level.cells[row, col + 1] is not None:
                            potential_directions.append(LineDirection.DOWN_RIGHT)
                    else:
                        if row < level.rows and level.cells[row, col + 1] is not None:
                            potential_directions.append(LineDirection.UP_RIGHT)
                        if row < level.rows - 1 and level.cells[row + 1, col + 1] is not None:
                            potential_directions.append(LineDirection.DOWN_RIGHT)
                if len(potential_directions) > 0:
                    line = check_line(screenshot=screenshot, level=level, x=x, y=y, row=row, col=col,
                                      potential_directions=potential_directions)
                    if line is not None:
                        lines.append(line)
    return lines


def find_blue_remaining(screenshot: np.ndarray):
    x = X_END
    y = Y_START
    while screenshot[y, x] != BLUE:
        x -= 1
        y += 1

    remaining_left_edge = find_extremity(screenshot=screenshot[y, :], start=x, limit=X_START, step=-1)
    remaining_right_edge = find_extremity(screenshot=screenshot[y, :], start=x, limit=X_END, step=1)
    remaining_top_edge = find_extremity(screenshot=screenshot[:, x], start=y, limit=Y_START, step=-1)
    remaining_bot_edge = find_extremity(screenshot=screenshot[:, x], start=y, limit=Y_END, step=1)

    y_temp = int((remaining_top_edge + 2 * remaining_bot_edge) / 3)
    shapes = list()
    for x_temp in range(remaining_right_edge, remaining_left_edge, -1):
        check_pixel_for_shape(screenshot=screenshot, x=x_temp, y=y_temp, shapes=shapes)

    blue_remaining = 0
    factor = 1
    for shape in shapes:
        blue_remaining += factor * identify_digit(shape=shape)
        factor *= 10

    return blue_remaining


def measure_cell(screenshot: np.ndarray, x: int, y: int):
    return HORIZONTAL_DISTANCE, VERTICAL_HALF_DISTANCE


def find_dimensions(screenshot: np.ndarray):
    left, _ = find_left_edge(screenshot)
    right, _ = find_right_edge(screenshot)
    top_x, top = find_top_edge(screenshot)
    bot_x, bot = find_bot_edge(screenshot)

    horizontal_distance, vertical_half_distance = measure_cell(screenshot=screenshot, x=top_x, y=top)

    cols = round((right - left) / horizontal_distance) + 1

    # In the representation every second column is moved up half a cell
    top_col = round((top_x - left) / horizontal_distance)
    if top_col % 2 == 1:
        top -= vertical_half_distance
    bot_col = round((bot_x - left) / horizontal_distance)
    if bot_col % 2 == 1:
        bot -= vertical_half_distance

    rows = round((bot - top) / (2 * vertical_half_distance)) + 1

    return left, right, top, bot, rows, cols, vertical_half_distance, horizontal_distance


def open_hexcells():
    hexcells_location = None
    for filename in os.listdir(LOGO_DIRECTORY):
        hexcells_location = pag.locateOnScreen(os.path.join(LOGO_DIRECTORY, filename))
        if hexcells_location is not None:
            break

    if hexcells_location is None:
        raise HexcellsLogoNotFoundException

    pag.click(pag.center(hexcells_location))


def grab_level(region=None):
    try:
        open_hexcells()

        pic = pag.screenshot(region=region).convert('L')
        # TODO: remove
        # pic.save('test9.png')
        return np.array(pic)
    except pag.ImageNotFoundException:
        print("Hexcells is not open")
        exit(1)


def parse(image=None):
    if image is None:
        screenshot = grab_level()
    else:
        screenshot = np.array(Image.open(image))
    left, right, top, bot, rows, cols, vertical_half_distance, horizontal_distance = find_dimensions(
        screenshot=screenshot)
    blue_remaining = find_blue_remaining(screenshot=screenshot)
    level = Level(left=left, right=right, top=top, bot=bot, rows=rows, cols=cols, blue_remaining=blue_remaining,
                  horizontal_distance=horizontal_distance, vertical_half_distance=vertical_half_distance)
    level.cells, level.orange_cells = init_cells(screenshot=screenshot, level=level)
    level.lines = init_lines(screenshot=screenshot, level=level)
    return level


class HexcellsLogoNotFoundException(Exception):
    pass
