import os
import multiprocessing

from mtrnn import Run


cwd = os.getcwd()
names = ['fixed_cst8', 'fixed_cst100', 'fixed_invexp',
         'learntau_cst8', 'learntau_cst100', 'learntau_invexp',
         'learntau_exp']

def run_training(cfg_name):
    os.chdir(cwd)
    run = Run.from_configfile('./exps/{}.yaml'.format(cfg_name))
    run.run()

pool = multiprocessing.Pool(processes=7)
for _ in pool.imap(run_training, names, chunksize=1):
    pass
