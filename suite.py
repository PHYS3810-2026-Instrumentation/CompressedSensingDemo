import os, sys, json, pathlib, datetime
import numpy as np
from glob import glob
import re

def operate_test(entry:dict, sequence:str='BSPF', logging:bool=False):
    _state_path = entry['_status']
    _kill_path = entry['_kill']

    def savestate(path, name, folder, sequence, iter, vars, cidx, fidx, pidx, sidx, bidx):
        tmp = {
            'name' : name,
            'folder' : folder,
            'sequence' : sequence,
            'iteration' : iter,
            'vars' : vars,
            'pointers' : [cidx, fidx, pidx, sidx, bidx]
        }

        if os.path.exists(path):
            if os.path.exists(f'{path}.bak'):
                pathlib.Path.unlink(f'{path}.bak')
            os.rename(path, f'{path}.bak')

        with open(path, mode='w') as fp:
            json.dump(tmp, fp, indent=None)

    def logprint(fileptr, string, term = '\n'):
        if fileptr is None:
            return
        print(f'[{datetime.datetime.now()}]\t{string}', end=term, file=fileptr)

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

    # order in which to check rollovers, instead of just negative order thru ptrs array
    inc_ptrs = [0, 0, 0, 0, CIDX]
    idx = 0
    for Ch in sequence:
        match Ch:
            case 'F':
                inc_ptrs[idx] = FIDX
            case 'P':
                inc_ptrs[idx] = PIDX
            case 'S':
                inc_ptrs[idx] = SIDX
            case 'B':
                inc_ptrs[idx] = BIDX
        idx += 1

    folder_parts = re.split(r'[\\/]+', folder)
    for n in range(1, len(folder_parts)+1):
        tmp_folder = '/'.join(folder_parts[0:n])
        if not os.path.exists(tmp_folder):
            os.mkdir(tmp_folder)

    FILE_PTR = None
    if logging:
        logpath = entry['_log']
        pathlib.Path.touch(logpath)
        FILE_PTR = open(file=logpath, mode='a')

    RESUME_FLAG = False
    ITER_N = 0
    VAR_VALS = {}
    if os.path.exists(_state_path):
        with open(_state_path) as fp:
            state = json.load(fp)

        if state['name'] != name or state['folder'] != folder:
            print(f'Error - state found for {name} @ {folder} doesn\'t seem to match!')
            exit(-4)

        ptrs = state['pointers']
        if 'sequence' in state:
            prev_seq = state['sequence']
        else:
            prev_seq = sequence

        if 'vars' in state:
            VAR_VALS = state['vars']

        if 'iteration' in state:
            ITER_N = state['iteration']

        if prev_seq != sequence:
            logprint(FILE_PTR, f'Warning: current sequence {sequence} does not match previous sequence {prev_seq}! Continuing anyway. I trust your judgement.')
        RESUME_FLAG = True

    if RESUME_FLAG:
        logprint(FILE_PTR, f'Resuming suite run.')
    else:
        logprint(FILE_PTR, f'Beginning suite run.')

    # eval all file names
    temp = []
    for path in files:
        fnames = glob(path)
        temp += fnames
    files = temp
    del temp

    logprint(FILE_PTR, f'Identified {len(files)} files:\n\t{'\n\t'.join(files)}')

    MAXES = [len(commands), len(files), len(patterns), len(samples), len(bases)]

    logprint(FILE_PTR, f'Maxes (SEQ={sequence}): {','.join([str(e) for e in MAXES])} ({np.prod(MAXES).astype(int)} combinations)')

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
        logprint(FILE_PTR, f'Starting command\t{COMMAND_RAW}')

        # check mono replacements
        for mono in MONO_REPLACE:
            if mono in COMMAND_RAW:
                COMMAND_RAW = COMMAND_RAW.replace(mono, MONO_REPLACE[mono])

        DYNAMIC_FLAGS = [False] * 5
        perms = 1
        pattern = r'\[[\w|\d|\=]*\]'
        dynamics = re.findall(pattern, COMMAND_RAW)
        MUTABLE_DYNAMICS = { }
        MUTABLE_VALUES = { }
        MUTABLE_ORDER = []
        for dyn in dynamics:
            if dyn in DYNAMIC_REPLACE:
                k = DYNAMIC_REPLACE[dyn][1]
                if not DYNAMIC_FLAGS[k]:
                    perms *= MAXES[k]
                DYNAMIC_FLAGS[k] = True
                if MAXES[k] == 0:
                    logprint(FILE_PTR, f'Error - Calling upon {dyn} in command {COMMAND_RAW}, but {dyn} has no entries!')
                    exit(-4)
            elif dyn in MONO_REPLACE:
                # should never hit
                pass
            else:
                rawdyn = dyn
                # strip off the leading and trailing brackets
                dyn = dyn[1:-1]
                parts = dyn.split('=')
                varname = parts[0]
                assignment = -1
                if len(parts) > 1:
                    assignment = int(parts[1])
                    COMMAND_RAW = COMMAND_RAW.replace(rawdyn, f'[{varname}]')
                if varname in MUTABLE_DYNAMICS:
                    if MUTABLE_DYNAMICS[varname] == -1 and assignment > 0:
                        MUTABLE_DYNAMICS[varname] = assignment
                        perms *= assignment
                else:
                    MUTABLE_DYNAMICS[varname] = assignment
                    if assignment is not None:
                        perms *= assignment
                    MUTABLE_ORDER.append(varname)

        for varname in MUTABLE_DYNAMICS:
            MUTABLE_VALUES[varname] = 0
            if MUTABLE_DYNAMICS[varname] < 0:
                logprint(FILE_PTR, f'Invalid value range for suite variable {varname} = {MUTABLE_DYNAMICS[varname]}. Possibly not defined.')
                MUTABLE_DYNAMICS[varname] = 0
            if varname in VAR_VALS:
                MUTABLE_VALUES[varname] = VAR_VALS[varname]

        VAR_VALS = { }
        logprint(FILE_PTR, f'Command should have {perms} permutation(s)')

        sent_flag = True
        N = ITER_N
        ITER_N = 0
        while sent_flag:
            savestate(_state_path, name, folder, sequence, N, MUTABLE_VALUES, *ptrs)

            if pathlib.Path.exists(_kill_path):
                # kill process
                logprint(FILE_PTR, f'Found a kill file! {_kill_path}. Halting here. Good luck to the next run!')
                if logging:
                    FILE_PTR.flush()
                    FILE_PTR.close()
                pathlib.Path.unlink(_kill_path)
                exit(0)
            
            # logprint(FILE_PTR, 'Updated state file')
            command = COMMAND_RAW
            params = []
            # insert elements into command
            for dyn in DYNAMIC_REPLACE:
                if dyn in command:
                    replace_set, replace_ptr, _ = DYNAMIC_REPLACE[dyn]
                    tmp = replace_set[ptrs[replace_ptr]]
                    command = command.replace(dyn, tmp)
                    params.append(f'{dyn}:{tmp}')

            for dyn in MUTABLE_VALUES:
                replstr = f'[{dyn}]'
                if f'[{dyn}]' in command:
                    tmp = str(MUTABLE_VALUES[dyn])
                    command = command.replace(replstr, tmp)
                    params.append(f'{replstr}:{tmp}')

            # issue command
            params = ', '.join(params)
            logprint(FILE_PTR, f'{N}/{perms}\t{params}')
            if logging:
                FILE_PTR.flush()
            response = cspy.cspy_engine(command)
            logprint(FILE_PTR, f'Code {response}')
            N += 1

            # first tick up vars
            submissive_anchor_inc = 0

            while submissive_anchor_inc < len(MUTABLE_ORDER):
                varname = MUTABLE_ORDER[submissive_anchor_inc]
                MUTABLE_VALUES[varname] = MUTABLE_VALUES[varname] + 1
                if MUTABLE_VALUES[varname] >= MUTABLE_DYNAMICS[varname]:
                    MUTABLE_VALUES[varname] = 0
                    submissive_anchor_inc += 1
                else:
                    break

            if submissive_anchor_inc < len(MUTABLE_ORDER):
                continue

            # then tick up counters
            super_anchor_inc = 0
            while super_anchor_inc < len(inc_ptrs):
                anchor_inc = inc_ptrs[super_anchor_inc]
                kill_flag = False
                if anchor_inc == 0:
                    ptrs[anchor_inc] += 1
                    sent_flag = False
                    break

                if not DYNAMIC_FLAGS[anchor_inc]:
                    super_anchor_inc += 1
                    continue
                else:
                    ptrs[anchor_inc] += 1
                    if ptrs[anchor_inc] >= MAXES[anchor_inc]:
                        ptrs[anchor_inc] = 0
                        super_anchor_inc += 1
                    else:
                        kill_flag = True

                if kill_flag:
                    break

    logprint(FILE_PTR, f'Finished suite! Wrapping up')
    if logging:
        FILE_PTR.flush()
        FILE_PTR.close()
    
    # clean up states
    if os.path.exists(_state_path):
        pathlib.Path.unlink(_state_path)
    if os.path.exists(_state_path + '.bak'):
        pathlib.Path.unlink(_state_path + '.bak')

if len(sys.argv) <= 1:
    print(f'Provide a path to a .json file (or files) defining the test runs.')
    exit(0)

FORCE_ANSWER = ''
SEQUENCE = 'BSPF'
DO_WRITE = False
KILL = False
paths = []
args = sys.argv[1:]
idx = 0
while idx < len(args):
    arg = args[idx]
    if arg in ('-y', '-n', '--yes', '--no'):
        FORCE_ANSWER = 'y' if arg in ('-y', '--yes') else 'n'
    elif arg in ('-k', '--kill'):
        KILL = True
    elif arg in ('-l', '--log'):
        DO_WRITE = True
    elif arg in ('-s', '--sequence'):
        idx += 1
        SEQUENCE = args[idx].upper()
        if len(SEQUENCE) != 4:
            print(f'Sequence arg must be only 4 characters long! Got {SEQUENCE}')
            exit(0)
        if re.match(r'[F|B|S|P]+', SEQUENCE) == None:
            print(f'Sequence {SEQUENCE} did not match regex check! Must consist of unique occurrences of F,B,S,P')
            exit(0)
    else:
        paths.append(arg)
    idx += 1

def find_state_path(path:str):
    parts = re.split(r'[\\/]+', path)
    fname = parts[-1]
    fname = f'.{fname}.state'
    return '/'.join(parts[:-1]) + '/' + fname

def find_log_path(path:str):
    parts = re.split(r'[\\/]+', path)
    fname = parts[-1]
    fname = f'.{fname}.log'
    return '/'.join(parts[:-1]) + '/' + fname

def find_kill_path(path:str):
    parts = re.split(r'[\\/]+', path)
    fname = parts[-1]
    fname = f'.{fname}.kill'
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
    entry['_log'] = find_log_path(path)
    entry['_kill'] = find_kill_path(path)
    logpath = entry['_log']
    statuspath = entry['_status']
    killpath = entry['_kill']

    if os.path.exists(statuspath) or os.path.exists(f'{statuspath}.bak'):
        if KILL:
            if not os.path.exists(killpath):
                pathlib.Path.touch(killpath)
            else:
                print(f'No status path exists for suite path {path}. Will not attempt to kill.')
            continue

        print(f'Found a file at status path {statuspath}: resume from this state for suite {path}? (Y/N)')
        usr = FORCE_ANSWER
        while usr not in ('y', 'n'):
            usr = input()
            usr = usr.lower()
        if usr == 'n':
            pathlib.Path.unlink(statuspath)
            if os.path.exists(f'{statuspath}.bak'):
                pathlib.Path.unlink(f'{statuspath}.bak')
            if os.path.exists(logpath):
                pathlib.Path.unlink(logpath)
        else:
            if not os.path.exists(statuspath):
                os.rename(f'{statuspath}.bak', statuspath)

    entries.append(entry)

for entry in entries:
    operate_test(entry, SEQUENCE, DO_WRITE)
