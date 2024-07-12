# Coherent-Rates

An investigation into the propagation of a coherent wavepacket.

## Building

Before building the devcontiner, first call

```
git submodule update --init --recursive
```

## Profiling

To profile use

```
python -m pip install flameprof
python -m cProfile -o myscript.prof examples/main.py
python -m flameprof  myscript.prof > output.svg
```
