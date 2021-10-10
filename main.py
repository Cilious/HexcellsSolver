from level_parser import parse, HexcellsLogoNotFoundException


try:
    print(parse("test2.png"))
except HexcellsLogoNotFoundException:
    print("Hexcells Logo was not found in the taskbar. Try:")
    print("     1. Ensure Hexcells is open")
    print("     2. Take a screenshot of the logo in the taskbar and add it to the directory hexcells_logo")

