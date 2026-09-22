import os, sys, json, pathlib
import numpy as np
from glob import glob
import re

def operate_test(entry:dict):
    _state_path = entry['_status']

    def savestate(path, name, folder, cidx, fidx, pidx, sidx, bidx):
        tmp = {
            'name' : name,
            'folder' : folder,
            'pointers' : [cidx, fidx, pidx, sidx, bidx]
        }

        if os.path.exists(path):
            if os.path.exists(f'{path}.bak'):
                pathlib.Path.unlink(f'{path}.bak')
            os.rename(path, f'{path}.bak')

        with open(path, mode='w') as fp:
            json.dump(tmp, fp, indent=None)

    name = entry['name']
    folder = entry['folder']
    commands = entry['commands']
    files = entry['files']
    patterns = entry['patterns']
    samples = entry['samples']
    bases = entry['bases']

    # cidx, fidx, pidx, sidx, bidx
    CIDX = 0
    FIDX = 1
    PIDX = 2
    SIDX = 3
    BIDX = 4
    ptrs = [0, 0, 0, 0, 0]

    folder_parts = re.split(r'[\\/]+', folder)
    for n in range(1, len(folder_parts)+1):
        tmp_folder = '/'.join(folder_parts[0:n])
        if not os.path.exists(tmp_folder):
            os.mkdir(tmp_folder)

    if os.path.exists(_state_path):
        with open(_state_path) as fp:
            state = json.load(fp)

        if state['name'] != name or state['folder'] != folder:
            print(f'Error - state found for {name} @ {folder} doesn\'t seem to match!')
            exit(-4)

        ptrs = state['pointers']
    

    # eval all file names

    temp = []
    for path in files:
        fnames = glob(path)
        temp += fnames
    files = temp
    del temp

    MAXES = [len(commands), len(files), len(patterns), len(samples), len(bases)]

    MONO_REPLACE = {
        '[FOLDER]' : folder,
        '[NAME]' : name
    }

    temp_num_set = [f'{e}' for e in range(len(files))]
    fname = ['.'.join((re.split(r'[\\/]+', e)[-1]).split('.')[:-1]) for e in files]

    DYNAMIC_REPLACE = {
        '[FILE]' : (files, FIDX, True),
        '[FILE-ID]' : (temp_num_set, FIDX, True),
        '[FNAME]' : (fname, FIDX, True),
        '[SAMPLE]' : (samples, SIDX, False),
        '[BASIS]' : (bases, BIDX, False),
        '[PATTERN]' : (patterns, PIDX, False)
    }

    # get the command parts, join it, then identify variables and flag them.
    import cspy
    while ptrs[CIDX] < MAXES[CIDX]:
        COMMAND_RAW = ', '.join(commands[ptrs[CIDX]])

        # check mono replacements
        for mono in MONO_REPLACE:
            if mono in COMMAND_RAW:
                COMMAND_RAW = COMMAND_RAW.replace(mono, MONO_REPLACE[mono])

        DYNAMIC_FLAGS = [False] * 5
        for dyn in DYNAMIC_REPLACE:
            if dyn in COMMAND_RAW:
                k = DYNAMIC_REPLACE[dyn][1]
                DYNAMIC_FLAGS[k] = True
                if MAXES[k] == 0:
                    print(f'Error - Calling upon {dyn} in command {COMMAND_RAW}, but {dyn} has no entries!')
                    exit(-4)

        sent_flag = True
        while sent_flag:
            savestate(_state_path, name, folder, *ptrs)
            command = COMMAND_RAW
            # insert elements into command
            for dyn in DYNAMIC_REPLACE:
                if dyn in command:
                    replace_set, replace_ptr, _ = DYNAMIC_REPLACE[dyn]
                    command = command.replace(dyn, replace_set[ptrs[replace_ptr]])

            # issue command
            cspy.cspy_engine(command)

            # tick up counters
            anchor_inc = BIDX
            while anchor_inc >= 0:
                kill_flag = False
                if anchor_inc == 0:
                    ptrs[anchor_inc] += 1
                    sent_flag = False

                if not DYNAMIC_FLAGS[anchor_inc]:
                    anchor_inc -= 1
                    continue
                else:
                    ptrs[anchor_inc] += 1
                    if ptrs[anchor_inc] >= MAXES[anchor_inc]:
                        ptrs[anchor_inc] = 0
                        anchor_inc -= 1
                    else:
                        kill_flag = True

                if kill_flag:
                    break

    # clean up states
    if os.path.exists(_state_path):
        pathlib.Path.unlink(_state_path)
    if os.path.exists(_state_path + '.bak'):
        pathlib.Path.unlink(_state_path + '.bak')

if len(sys.argv) <= 1:
    print(f'Provide a path to a .json file (or files) defining the test runs.')
    exit(0)

paths = sys.argv[1:]

def find_state_path(path:str):
    parts = re.split(r'[\\/]+', path)
    fname = parts[-1]
    fname = f'.{fname}.state'
    return '/'.join(parts[:-1]) + '/' + fname

entries = []

for path in paths:
    if not os.path.exists(path):
        print(f'File {path} not found.')
        exit(-1)
    if not path.endswith('.json'):
        print(f'File {path} is not a .json file')
        exit(-2)

    try:
        with open(path, mode='r') as fp:
            entry = json.load(fp)
    except Exception as e:
        print(f'Error while loading file {path}:\n{e}')
        exit(-3)

    entry['_status'] = find_state_path(path)
    statuspath = entry['_status']

    if os.path.exists(statuspath) or os.path.exists(f'{statuspath}.bak'):
        print(f'Found a file at status path {statuspath}: resume from this state for suite {path}? (Y/N)')
        usr = ''
        while usr not in ('y', 'n'):
            usr = input()
            usr = usr.lower()
        if usr == 'n':
            pathlib.Path.unlink(statuspath)
            if os.path.exists(f'{statuspath}.bak'):
                pathlib.Path.unlink(f'{statuspath}.bak')
        else:
            if not os.path.exists(statuspath):
                os.rename(f'{statuspath}.bak', statuspath)

    entries.append(entry)

for entry in entries:
    operate_test(entry)
