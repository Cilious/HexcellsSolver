import traceback
from pyautogui import click, moveTo
from time import sleep

from level_parser import parse, HexcellsLogoNotFoundException
from solver import solve, NoSolutionFoundException

AFTER_LEVEL_SLEEP = 2
MENU_X = 955
MENU_Y = 883
AFTER_MENU_SLEEP = 2
RANDOM_SEED_X = 535
RANDOM_SEED_Y = 525
AFTER_RANDOM_SLEEP = 0.2
GENERATE_X = 955
GENERATE_Y = 779
AFTER_GENERATE_SLEEP = 6


try:
    first_run = True
    while True:
        level = parse(opened=not first_run)
        solve(level)
        sleep(AFTER_LEVEL_SLEEP)
        click(x=MENU_X, y=MENU_Y)
        sleep(AFTER_MENU_SLEEP)
        click(x=RANDOM_SEED_X, y=RANDOM_SEED_Y)
        sleep(AFTER_RANDOM_SLEEP)
        click(x=GENERATE_X, y=GENERATE_Y)
        moveTo(x=1, y=1)
        sleep(AFTER_GENERATE_SLEEP)
        first_run = False
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
