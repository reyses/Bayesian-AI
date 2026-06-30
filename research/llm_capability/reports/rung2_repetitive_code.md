# RUNG 2 — repetitive code under no-magic-numbers rule | model=gemma4:latest | n=4

            spec | compiles | no-magic | note
----------------------------------------------------------------
 dataclass field |        Y |        Y | clean
    converter fn |        Y |        Y | clean
    enum mapping |        Y |        Y | clean
    rolling stat |        n |        - | SyntaxError: unterminated triple-quoted str

COMPILES: 3/4 = 75%
OBEYS no-magic-numbers (of those that compile): 3/3
latency/call: mean 10.1s
