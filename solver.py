import pyautogui as pag
import numpy as np
from pysat.solvers import Minisat22
from itertools import combinations

from time import sleep

from representation import Level, CellType, Line, ConnectivityType
from level_parser import identify_cell

BLUE_CLICK = 'right'
GRAY_CLICK = 'left'


def trace_line(level: Level, line: Line):
    index_change = [[(0, 1), (1, 0), (0, -1), (-1, -1), (-1, 0), (-1, 1)],
                    [(1, 1), (1, 0), (1, -1), (0, -1), (-1, 0), (0, 1)]]
    row = line.start_row
    col = line.start_col
    cells = list()
    section = list()
    while 0 <= row < level.rows and 0 <= col < level.cols:
        if level.cells[row, col] is not None:
            if level.cells[row, col].cell_type == CellType.GRAY:
                if len(section) > 0:
                    cells.append(section)
                    section = list()
            else:
                section.append((row, col))
        change = index_change[col % 2][line.direction.value]
        row += change[0]
        col += change[1]

    if len(section) > 0:
        cells.append(section)

    return cells


def get_blue_cell_neighbours(level: Level, row: int, col: int):
    neighbours = list()
    independent_indices = [(-2, 0), (-1, 0), (1, 0), (2, 0), (-1, -2), (0, -2), (1, -2), (-1, 2), (0, 2), (1, 2)]
    dependent_indices = [[(r, -1) for r in range(-2, 2)] + [(r, 1) for r in range(-2, 2)],
                         [(r, -1) for r in range(-1, 3)] + [(r, 1) for r in range(-1, 3)]]

    for cell_index in independent_indices + dependent_indices[col % 2]:
        cell_row = row + cell_index[0]
        cell_col = col + cell_index[1]
        if 0 <= cell_row < level.rows and 0 <= cell_col < level.cols \
                and level.cells[cell_row, cell_col] is not None \
                and level.cells[cell_row, cell_col].cell_type != CellType.GRAY:
            neighbours.append((cell_row, cell_col))

    return [neighbours]


def get_gray_cell_neighbours(level: Level, row: int, col: int):
    neighbour_indices = [[(-1, 0), (-1, 1), (0, 1), (1, 0), (0, -1), (-1, -1)],
                         [(-1, 0), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]]

    neighbours: list[list[(int, int)]] = list()
    section = list()
    first_is_none = False
    for cell_index in neighbour_indices[col % 2]:
        cell_row = row + cell_index[0]
        cell_col = col + cell_index[1]
        if 0 <= cell_row < level.rows and 0 <= cell_col < level.cols and \
                level.cells[cell_row, cell_col] is not None and \
                level.cells[cell_row, cell_col].cell_type != CellType.GRAY:
            section.append((cell_row, cell_col))
        else:
            if len(section) > 0:
                neighbours.append(section)
                section = list()
            if cell_index == neighbour_indices[col % 2][0]:
                first_is_none = True
    if len(section) > 0:
        if len(neighbours) == 0 or first_is_none:
            neighbours.append(section)
        else:
            neighbours[0] = section + neighbours[0]

    return neighbours


def apply_solutions(level: Level, blue_cells: list[(int, int)], gray_cells: list[(int, int)],
                    informative_cells: dict[(int, int), list[(int, int)]]):
    for blue_cell in blue_cells:
        x, y = level.cell_coordinates(row=blue_cell[0], col=blue_cell[1])
        pag.click(x=x, y=y, button=BLUE_CLICK)
    for gray_cell in gray_cells:
        x, y = level.cell_coordinates(row=gray_cell[0], col=gray_cell[1])
        pag.click(x=x, y=y, button=GRAY_CLICK)

    if len(blue_cells) + len(gray_cells) < level.orange_cells:  # TODO: look into problem with success screen
        sleep(1)
        screenshot = np.array(pag.screenshot().convert('L'))
        for blue_cell in blue_cells:
            x, y = level.cell_coordinates(row=blue_cell[0], col=blue_cell[1])
            level.cells[blue_cell[0], blue_cell[1]] = identify_cell(screenshot=screenshot, x=x, y=y)
            if level.cells[blue_cell[0], blue_cell[1]].number is not None:
                neighbours = get_blue_cell_neighbours(level=level, row=blue_cell[0], col=blue_cell[1])
                if len(neighbours) > 0:
                    informative_cells[blue_cell] = neighbours
        for gray_cell in gray_cells:
            x, y = level.cell_coordinates(row=gray_cell[0], col=gray_cell[1])
            level.cells[gray_cell[0], gray_cell[1]] = identify_cell(screenshot=screenshot, x=x, y=y)
            if level.cells[gray_cell[0], gray_cell[1]].number is not None:
                neighbours = get_gray_cell_neighbours(level=level, row=gray_cell[0], col=gray_cell[1])
                if len(neighbours) > 0:
                    informative_cells[gray_cell] = neighbours

    level.blue_remaining -= len(blue_cells)
    level.orange_cells -= (len(blue_cells) + len(gray_cells))


def distribute(clauses: list[list[int]]):
    if len(clauses) == 0:
        return []
    elif len(clauses) == 1:
        return [[cell] for cell in clauses[0]]
    else:
        result = list()
        for cell in clauses[0]:
            for clause in distribute(clauses=clauses[1:]):
                result.append(clause + [cell])
        return result


def n_out_of_list(cells: list[(int, int)], n: int):
    cell_codes = [to_cell_code(c) for c in cells]
    positive_subsets = [list(clause) for clause in combinations(cell_codes, len(cell_codes) - n + 1)]
    if n * 2 == len(cell_codes):
        negative_subsets = [[-c for c in subset] for subset in positive_subsets]
    else:
        negative_subsets = [[-c for c in clause] for clause in combinations(cell_codes, n + 1)]
    return negative_subsets + positive_subsets


def from_cell_code(code: int):
    return (code - 1) / 100, (code - 1) % 100


def to_cell_code(row_col: (int, int)):
    return int(row_col[0] * 100 + row_col[1] + 1)


def find_sat_solutions(solver: Minisat22, cell_dependencies: dict[(int, int), int]):
    if not solver.solve():
        raise NoSolutionFoundException("Contradiction in board.")  # 11873979

    blue_cells = list()
    gray_cells = list()

    cells = sorted(cell_dependencies.keys(), key=cell_dependencies.get, reverse=True)
    found_solution = True
    while found_solution:
        found_solution = False
        for cell in cells:
            cell_code = to_cell_code(cell)
            if not solver.solve(assumptions=[-cell_code]):
                blue_cells.append(cell)
                solver.add_clause([cell_code])
                cells.remove(cell)
                found_solution = True
                break
            elif not solver.solve(assumptions=[cell_code]):
                gray_cells.append(cell)
                solver.add_clause([-cell_code])
                cells.remove(cell)
                found_solution = True
                break

    return blue_cells, gray_cells


def find_total_remaining_solutions(level: Level, solver: Minisat22, cell_dependencies: dict[(int, int), int]):
    orange_cells = list()
    for row in range(level.rows):
        for col in range(level.cols):
            if level.cells[row, col] is not None and level.cells[row, col].cell_type == CellType.ORANGE:
                orange_cells.append((row, col))

    if level.orange_cells == level.blue_remaining:
        return orange_cells, [], True
    elif level.blue_remaining == 0:
        return [], orange_cells, True

    solver.append_formula(n_out_of_list(orange_cells, level.blue_remaining))
    for cell in orange_cells:
        cell_dependencies[cell] = cell_dependencies.get(cell, 0) + 1

    return [], [], False


def find_disjointed_solutions(level: Level, line_or_cell, cells: list[list[(int, int)]],
                              solver: Minisat22, cell_dependencies: dict[(int, int), int]):
    blue_cells = list()
    gray_cells = list()
    done = False

    orange_cells = list()
    # -2: Multiple sections with blue cell(s), -1: No section with blue cell(s), x: Section x contains blue cell(s)
    blue_section = -1
    number_of_blues = 0
    dangerous_sections = list()
    for section_index, section in enumerate(cells):
        if len(section) >= line_or_cell.number:
            dangerous_sections.append(section_index)
        for cell in section:
            if level.cells[cell[0], cell[1]].cell_type == CellType.ORANGE:
                orange_cells.append(cell)
            else:
                number_of_blues += 1
                if blue_section == -1:
                    blue_section = section_index
                elif blue_section != section_index:
                    blue_section = -2

    if number_of_blues == line_or_cell.number:
        gray_cells = orange_cells
        for cell in orange_cells:
            solver.add_clause([-to_cell_code(cell)])
        done = True
    elif len(orange_cells) == line_or_cell.number - number_of_blues:
        blue_cells = orange_cells
        for cell in orange_cells:
            solver.add_clause([to_cell_code(cell)])
        done = True
    else:
        solver.append_formula(formula=n_out_of_list(orange_cells, line_or_cell.number - number_of_blues))
        if blue_section >= 0:
            dangerous_sections = [blue_section]
        for section_index in dangerous_sections:
            section = cells[section_index]
            for start in range(len(section) - line_or_cell.number + 1):
                clause = list()
                for cell in section[start:start + line_or_cell.number]:
                    if level.cells[cell[0], cell[1]].cell_type == CellType.ORANGE:
                        clause.append(-to_cell_code(cell))
                if len(clause) > 0:
                    solver.add_clause(clause=clause)
        for cell in orange_cells:
            cell_dependencies[cell] = cell_dependencies.get(cell, 0) + 1

    return blue_cells, gray_cells, done


def find_connected_solutions(level: Level, line_or_cell, cells: list[list[(int, int)]],
                             solver: Minisat22, cell_dependencies: dict[(int, int), int]):
    blue_cells = list()
    gray_cells = list()
    done = False

    gray_sections = list()
    known_blues = list()
    for section_index, connected_cells in enumerate(cells):
        if len(connected_cells) < line_or_cell.number:
            gray_sections.append(section_index)
        else:
            for cell_index, cell in enumerate(cells[0]):
                if level.cells[cell[0], cell[1]].cell_type == CellType.BLUE:
                    known_blues.append(cell_index)
        if len(known_blues) > 0:
            gray_sections = list(range(section_index)) + list(range(section_index + 1, len(cells)))

    for section_index in reversed(gray_sections):
        gray_cells += cells[section_index]
        for cell in cells[section_index]:
            solver.add_clause([-to_cell_code(cell)])
        del cells[section_index]

    if len(known_blues) > 0:
        section = cells[0]
        blue_start = min(known_blues[0], len(section) - line_or_cell.number)
        blue_end = max(known_blues[len(known_blues) - 1], line_or_cell.number - 1)
        for cell in section[blue_start: blue_end]:
            if level.cells[cell[0], cell[1]].cell_type == CellType.ORANGE:
                blue_cells.append(cell)
                solver.add_clause([to_cell_code(cell)])
        if blue_end - blue_start + 1 == line_or_cell.number:
            for cell in section[0:blue_start]:
                gray_cells.append(cell)
                solver.add_clause([-to_cell_code(cell)])
            for cell in section[blue_end + 1: len(section)]:
                gray_cells.append(cell)
                solver.add_clause([-to_cell_code(cell)])
            done = True
        else:
            possible_blues = section[blue_end - line_or_cell.number + 1:blue_start] + \
                             section[blue_end + 1: blue_start + line_or_cell.number]
            remaining_blue = line_or_cell.number - (blue_end - blue_start + 1)
            clauses = list()
            for start in range(len(possible_blues) - remaining_blue + 1):
                clause = list()
                for cell in possible_blues[0:start]:
                    clause.append(-to_cell_code(cell))
                for cell in possible_blues[start:start + remaining_blue]:
                    clause.append(to_cell_code(cell))
                for cell in possible_blues[start + remaining_blue:len(possible_blues)]:
                    clause.append(-to_cell_code(cell))
                clauses.append(clause)
            solver.append_formula(distribute(clauses=clauses))
            for cell in section[0:blue_end - line_or_cell.number + 1]:
                gray_cells.append(cell)
                solver.add_clause([-to_cell_code(cell)])
            for cell in section[blue_start + line_or_cell.number + 1:len(section)]:
                gray_cells.append(cell)
                solver.add_clause([-to_cell_code(cell)])
            for cell in possible_blues:
                cell_dependencies[cell] = cell_dependencies.get(cell, 0) + 1
    else:
        clauses = list()
        for section_index, section in enumerate(cells):
            clause_template = list()
            for other_section in cells[:section_index] + cells[section_index + 1:]:
                for cell in other_section:
                    clause_template.append(-to_cell_code(cell))
            for start in range(len(section) - line_or_cell.number + 1):
                clause = clause_template.copy()
                for cell in section[start:start + line_or_cell.number]:
                    clause.append(to_cell_code(cell))
                clauses.append(clause)
            for cell in section:
                cell_dependencies[cell] = cell_dependencies.get(cell, 0) + 1
        solver.append_formula(distribute(clauses))

    return blue_cells, gray_cells, done


def find_arbitrary_solutions(level: Level, cell_or_line, sections: list[list[(int, int)]],
                             solver: Minisat22, cell_dependencies: dict[(int, int), int]):
    blue_cells = list()
    gray_cells = list()
    done = False

    orange_cells = list()
    number_of_blues = 0
    for section in sections:
        for cell in section:
            if level.cells[cell[0], cell[1]].cell_type == CellType.ORANGE:
                orange_cells.append(cell)
            elif level.cells[cell[0], cell[1]].cell_type == CellType.BLUE:
                number_of_blues += 1

    if number_of_blues == cell_or_line.number:
        gray_cells = orange_cells
        for cell in orange_cells:
            solver.add_clause([-to_cell_code(cell)])
        done = True
    elif len(orange_cells) == cell_or_line.number - number_of_blues:
        blue_cells = orange_cells
        for cell in orange_cells:
            solver.add_clause([to_cell_code(cell)])
        done = True
    else:
        solver.append_formula(n_out_of_list(orange_cells, cell_or_line.number - number_of_blues))
        for cell in orange_cells:
            cell_dependencies[cell] = cell_dependencies.get(cell, 0) + 1

    return blue_cells, gray_cells, done


def find_disjointed_cell_solutions(level: Level, cell: (int, int), neighbours: list[list[(int, int)]],
                                   solver: Minisat22, cell_dependencies: dict[(int, int), int]):
    if len(neighbours[0]) < 6:
        return find_disjointed_solutions(level=level, line_or_cell=level.cells[cell[0], cell[1]],
                                         cells=neighbours, solver=solver, cell_dependencies=cell_dependencies)

    orange_cells = list()
    blue_cells = list()
    for cell_index in neighbours[0]:
        if level.cells[cell_index[0], cell_index[1]].cell_type == CellType.BLUE:
            blue_cells.append(cell_index)
        else:
            orange_cells.append(cell_index)

    number = level.cells[cell[0], cell[1]].number
    if len(blue_cells) == number:
        return [], orange_cells, True

    solver.append_formula(formula=n_out_of_list(orange_cells, number - len(blue_cells)))
    section = neighbours[0] + neighbours[0][:5]
    for start in range(6):
        clause = list()
        for cell_index in section[start:start+number]:
            if cell_index not in blue_cells:
                clause.append(-to_cell_code(cell_index))
        if len(clause) > 0:
            solver.add_clause(clause=clause)
    for cell_index in orange_cells:
        cell_dependencies[cell_index] = cell_dependencies.get(cell_index, 0) + 1
    return [], [], False


def find_connected_cell_solutions(level: Level, cell: (int, int), neighbours: list[list[(int, int)]],
                                  solver: Minisat22, cell_dependencies: dict[(int, int), int]):
    if len(neighbours[0]) < 6:
        return find_connected_solutions(level=level, line_or_cell=level.cells[cell[0], cell[1]],
                                        cells=neighbours, solver=solver, cell_dependencies=cell_dependencies)

    orange_cells = list()
    blue_cells = list()
    for cell_index in neighbours[0]:
        if level.cells[cell_index[0], cell_index[1]].cell_type == CellType.BLUE:
            blue_cells.append(cell_index)
        else:
            orange_cells.append(cell_index)

    number = level.cells[cell[0], cell[1]].number
    if number == len(blue_cells):
        return [], orange_cells, True
    if number - len(blue_cells) == len(orange_cells):
        return orange_cells, [], True

    section = neighbours[0] + neighbours[0][:5]
    clauses = list()
    for start in range(6):
        clause = list()
        legal_clause = True
        for cell_indices in section[start + number:start + 6]:
            if cell_indices in blue_cells:
                legal_clause = False
                break
            clause.append(-to_cell_code(cell_indices))
        if legal_clause:
            for cell_indices in section[start:start + number]:
                if cell_indices not in blue_cells:
                    clause.append(to_cell_code(cell_indices))
            clauses.append(clause)
    solver.append_formula(formula=distribute(clauses=clauses))
    for cell_indices in orange_cells:
        cell_dependencies[cell_indices] = cell_dependencies.get(cell_indices, 0) + 1
    return [], [], False


def find_solutions(level: Level, informative_cells: dict[(int, int), list[list[(int, int)]]],
                   informative_lines: dict[Line, list[list[(int, int)]]]):
    ms = Minisat22()
    cell_dependencies: dict[(int, int), int] = dict()

    blue_cells = list()
    gray_cells = list()
    for cell, neighbours in informative_cells.copy().items():  # TODO Check whether its necessary
        for section in neighbours:
            for neighbour in section:
                if level.cells[neighbour[0], neighbour[1]].cell_type == CellType.GRAY:
                    if level.cells[cell[0], cell[1]].cell_type == CellType.BLUE:
                        neighbours = get_blue_cell_neighbours(level=level, row=cell[0], col=cell[1])
                    else:
                        neighbours = get_gray_cell_neighbours(level=level, row=cell[0], col=cell[1])
                    if len(neighbours) > 0:
                        informative_cells[cell] = neighbours
                    else:
                        del informative_cells[cell]
                    break

        if level.cells[cell[0], cell[1]].connectivity_type == ConnectivityType.UNKNOWN:
            b, g, done = find_arbitrary_solutions(level=level, cell_or_line=level.cells[cell[0], cell[1]],
                                                  sections=neighbours, solver=ms,
                                                  cell_dependencies=cell_dependencies)
        elif level.cells[cell[0], cell[1]].connectivity_type == ConnectivityType.DISJOINTED:
            b, g, done = find_disjointed_cell_solutions(level=level, cell=cell, neighbours=neighbours,
                                                        solver=ms, cell_dependencies=cell_dependencies)
        else:
            b, g, done = find_connected_cell_solutions(level=level, cell=cell, neighbours=neighbours,
                                                       solver=ms, cell_dependencies=cell_dependencies)
        blue_cells += b
        gray_cells += g
        if done:
            del informative_cells[cell]

    for line, cells in informative_lines.copy().items():
        for section in cells:
            for cell in section:
                if level.cells[cell[0], cell[1]].cell_type == CellType.GRAY:
                    cells = trace_line(level=level, line=line)
                    if len(cells) > 0:
                        informative_lines[line] = cells
                    else:
                        del informative_lines[line]
                    break

        if line.connectivity_type == ConnectivityType.UNKNOWN:
            b, g, done = find_arbitrary_solutions(level=level, cell_or_line=line, sections=cells,
                                                  solver=ms, cell_dependencies=cell_dependencies)
        elif line.connectivity_type == ConnectivityType.DISJOINTED:
            b, g, done = find_disjointed_solutions(level=level, line_or_cell=line, cells=cells,
                                                   solver=ms, cell_dependencies=cell_dependencies)
        else:
            b, g, done = find_connected_solutions(level=level, line_or_cell=line, cells=cells,
                                                  solver=ms, cell_dependencies=cell_dependencies)
        blue_cells += b
        gray_cells += g
        if done:
            del informative_lines[line]

    for cell in blue_cells + gray_cells:
        if cell in cell_dependencies:
            del cell_dependencies[cell]

    b, g = find_sat_solutions(solver=ms, cell_dependencies=cell_dependencies)
    blue_cells += b
    gray_cells += g

    if len(blue_cells) == 0 and len(gray_cells) == 0:
        b, g, _ = find_total_remaining_solutions(level=level, solver=ms, cell_dependencies=cell_dependencies)
        if len(b) == len(g) == 0:
            b, g = find_sat_solutions(solver=ms, cell_dependencies=cell_dependencies)
        blue_cells = b
        gray_cells = g

    blue_cells = [tuple(c) for c in np.unique(blue_cells, axis=0)]
    gray_cells = [tuple(c) for c in np.unique(gray_cells, axis=0)]

    ms.delete()

    return blue_cells, gray_cells


def pre_analyse(level: Level):
    informative_cells = dict()
    for row in range(level.rows):
        for col in range(level.cols):
            if level.cells[row, col] is not None and level.cells[row, col].number is not None:
                if level.cells[row, col].cell_type == CellType.GRAY:
                    neighbours = get_gray_cell_neighbours(level=level, row=row, col=col)
                    if len(neighbours) > 0:
                        informative_cells[(row, col)] = neighbours
                elif level.cells[row, col].cell_type == CellType.BLUE:
                    neighbours = get_blue_cell_neighbours(level=level, row=row, col=col)
                    if len(neighbours) > 0:
                        informative_cells[(row, col)] = neighbours

    informative_lines = dict()
    for line in level.lines:
        informative_lines[line] = trace_line(level=level, line=line)

    return informative_cells, informative_lines


def solve(level: Level):
    informative_cells, informative_lines = pre_analyse(level=level)

    while level.orange_cells > 0:
        blue_cells, gray_cells = find_solutions(level=level,
                                                informative_cells=informative_cells,
                                                informative_lines=informative_lines)
        if len(blue_cells) == 0 and len(gray_cells) == 0:
            raise NoSolutionFoundException("Not enough information.")

        apply_solutions(level=level, blue_cells=blue_cells, gray_cells=gray_cells, informative_cells=informative_cells)


class NoSolutionFoundException(Exception):
    pass
