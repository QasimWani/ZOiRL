from IPython.display import clear_output
import os
import datetime


clear = lambda: os.system("clear")  # for terminal


class flush:
    def __init__(self, file: str):
        self.file = file
        self.has_cleared_file = False

    def run_once(self):
        if not self.has_cleared_file and os.path.exists(self.file):
            os.remove(self.file)
            f = open(self.file, "a")
            date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            data = "Running session (YY-MM-DD H:m): " + str(date)
            f.write(data + "\n------------------------------\n")
            f.close()
            self.has_cleared_file = True


### Prints details and error logs.
file = "log.txt"
sys = flush(file)


def LOG(data: str, pprint: bool = True):

    sys.run_once()

    f = open(file, "a")
    f.write(data + "\n")
    f.close()

    if pprint:
        print("\r" + data, end="")
        # clear_output(wait=True)
        clear()
