from level_parser import parse, HexcellsLogoNotFoundException


"""
level = sc.grab_level()

x, y = sc.find_right_edge(level)
print(x, y)
pag.moveTo(x, y)
"""

try:
    print(parse())
except HexcellsLogoNotFoundException:
    print("Hexcells Logo was not found in the taskbar. Try:")
    print("     1. Ensure Hexcells is open")
    print("     2. Take a screenshot of the logo in the taskbar and add it to the directory hexcells_logo")

"""
orange:  185
gray:  62
blue:  125
background:  43
horizontal distance: 55
vertical half distance: 31
vertical full distance: 62
"""
