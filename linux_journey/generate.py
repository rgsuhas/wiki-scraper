def quux(x: int, /, *, y: int) -> None:
    pass

quux(3, y=5)  # Ok
#quux(x=3, 5)  # error: Too many positional arguments for "quux"
#quux(3, y=5)  # error: Unexpected keyword argument "x" for "quux