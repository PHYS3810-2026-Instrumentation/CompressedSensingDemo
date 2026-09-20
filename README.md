# CompressedSensingSuites

Python CLI for reconstruction of compressed sensing (random sampling) data. Operates both as a testing suite as well as a practical tool for the application.
Included is the file `suite.py`, which interprets .json files defining repetitive commands to perform.

Requires Python 3.14, and:
 - numpy
 - scipy
 - matplotlib
 - cvxpy
 - pywavelets
 - sklearn

## Basic Usage
The program `cspy.py` implements compressed sensing reconstruction as well as functions that may be useful in processing, such as image cropping & resizing, vectorization and devectorization of images, and generating sample masks. There are five main functions in `cspy.py`, all of which are specified as command line arguments:
    - prep
    - vector
    - sample
    - reconstruct
    - devector

`cspy.py` can be run via the command line (terminal, command prompt, powershell) with Python. Specifying any of these functions will display useful information such as arguments and syntax.

```
py cspy.py prep ...
py cspy.py vector ...
```

In the command line, multiple functions can be written with a comma separating each one (i.e. `py cspy.py prep ... , vector ...`). This operates a command pipeline, where the output of one function is passed directly into the other - this prevents the need to provide input paths for *all* functions in a pipeline (i.e. the output image of the `prep` function will be piped directly into the `vector` function as the input image). 

`cspy.py` can also be used as a library within another Python codebase. Each of the functions can be called independently, though the core function `cspy_engine` can be called with an argument string as one of the parameters and the `pipe` flag set to True, which will incentivize the function to return the final state of the command pipeline given.

## Suites

The file `suite.py` is another command-line utility capable of performing compressed sensing functions (those defined within `cspy.py`) upon numerous files or setups. `suite.py` takes a .json file defining the suite commands and parameters, and iteratively performs each command for all permutations of those command parameters. An example suite file can be seen in `./suites/multipattern.json`.

The suite file has five key fields, all of which are list objects. These fields are:
    - commands
    - files
    - patterns
    - samples
    - bases
The `commands` field is a list of lists of strings, where each sub-list defines multiple `cspy.py` function commands to be executed within a command pipeline. Each sub-list is disconnected from each other (i.e. are not operated as one long command pipeline). The commands can refer to parameters defined elsewhere in the suite file with square brackets. The list of parameters are below.
    - `[FILE]`      the path to each input file
    - `[FILE-ID]`   the ID of each input file
    - `[FNAME]`     the name of each input file, excluding path and filetype
    - `[BASIS]`     each transformation basis to be used
    - `[SAMPLE]`    each sample percentage to be used
    - `[PATTERN]`   each vectorization pattern to be used

The parameters are appropriately defined by the other four field lists. Each of these parameter field lists should be a list of strings.

When running, each command is executed in the order they are defined. Each command is evaluated once for each combination of parameters. 
`suite.py` records the progress of the overall command suite as it goes. If it is interrupted, attempting to run the same suite file again will prompt the user if they would like to resume from its prior state. Depending on the users choice it will either resume from the latest command permutation reached, or restart the entire process. Note that sub-command progress is not restored (i.e. if interrupted midway through a command pipeline, it will have to repeat that whole command pipeline).