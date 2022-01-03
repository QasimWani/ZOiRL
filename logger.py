from IPython.display import clear_output
import os


clear = lambda: os.system("cls")  # for terminal

### Prints details and error logs.
def LOG(data: str, pprint: bool = True, file: str = "log.txt"):
    f = open(file, "a")
    f.write(data + "\n")
    f.close()

    if pprint:
        print("\r" + data, end="")
        # clear_output(wait=True)
        clear()
