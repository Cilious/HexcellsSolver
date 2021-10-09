from enum import Enum
from numpy import array


class ConnectivityType(Enum):
    CONNECTED = 0
    DISJOINTED = 1
    UNKNOWN = 2


class CellType(Enum):
    ORANGE = 0
    BLUE = 1
    GRAY = 2


class Cell:
    def __init__(self, cell_type: CellType, connectivity_type: ConnectivityType, number: int = None):
        self.cell_type = cell_type
        self.number = number
        self.connectivity_type = connectivity_type

    def __str__(self):
        return f"{['O', 'B', 'G', 'N'][self.cell_type.value]}" \
               f"{self.number if self.number is not None else '?'}" \
               f"{['C', 'D', 'U'][self.connectivity_type.value]}"


class LineDirection(Enum):
    DOWN_RIGHT = 0
    DOWN = 1
    DOWN_LEFT = 2
    UP_LEFT = 3
    UP = 4
    UP_RIGHT = 5


class Line:
    def __init__(self, start_row: int, start_col: int, number: int,
                 connectivity_type: ConnectivityType, direction: LineDirection):
        self.start_row = start_row
        self.start_col = start_col
        self.number = number
        self.connectivity_type = connectivity_type
        self.direction = direction

    def __str__(self):
        return f"{self.start_row}," \
               f"{self.start_col}," \
               f"{self.number}," \
               f"{['C', 'D', 'U'][self.connectivity_type.value]}," \
               f"{['DR', 'D', 'DL', 'UL', 'U', 'UR'][self.direction.value]}"


class Level:
    def __init__(self, left: int, right: int, top: int, bot: int, rows: int, cols: int, blue_remaining: int,
                 horizontal_distance: int, vertical_full_distance: int, vertical_half_distance,
                 gray_cells: int = 0, cells: array = None, lines: list[Line] = None):
        self.left = left
        self.right = right
        self.top = top
        self.bot = bot
        self.rows = rows
        self.cols = cols
        self.blue_remaining = blue_remaining
        self.horizontal_distance = horizontal_distance
        self.vertical_full_distance = vertical_full_distance
        self.vertical_half_distance = vertical_half_distance
        self.gray_cells = gray_cells
        self.cells = cells
        self.lines = lines

    def cell_coordinates(self, row: int, col: int):
        x = self.left + self.horizontal_distance * col
        y = self.top + self.vertical_full_distance * row
        if col % 2 == 1:
            y += self.vertical_half_distance
        return x, y

    def __str__(self):
        meta_data = f"left={self.left}, right={self.right}, top={self.top}, bot={self.bot}, rows={self.rows}, " \
                    f"cols={self.cols}, remaining={self.blue_remaining}, gray_cells={self.gray_cells}\n"

        cells = ""
        for row in range(self.rows):
            for col in range(self.cols):
                if self.cells[row, col] is not None:
                    cells += str(self.cells[row][col]) + " "
                else:
                    cells += "N/A "
            cells += "\n"

        lines = ""
        for line in self.lines:
            lines += str(line) + "; "

        return meta_data + "\n" + cells + "\n" + lines
