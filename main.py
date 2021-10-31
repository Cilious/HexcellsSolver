import traceback

from level_parser import parse, HexcellsLogoNotFoundException
from solver import solve, NoSolutionFoundException


try:
    level = parse()
    print(level)
    solve(level)
except HexcellsLogoNotFoundException as e:
    print("Hexcells Logo was not found in the taskbar. Try:")
    print("     1. Ensure Hexcells is open")
    print("     2. Take a screenshot of the logo in the taskbar and add it to the directory hexcells_logo")
    print(e)
except NoSolutionFoundException as e:
    print("No solution was found.")
    print("Please contact the project creator with the seed of level.")
    print(e)
    traceback.print_exc()
except IndexError as e:
    print("Something went wrong. Please try again.")
    print("If the issue keeps persisting, please contact the project creator with the seed of level.")
    print(e)
    traceback.print_exc()
