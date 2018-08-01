import sys
from mtrnn import Run

run = Run.from_configfile(sys.argv[1])
run.run()
run.save()
