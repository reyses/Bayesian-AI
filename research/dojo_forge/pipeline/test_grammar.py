import sys
from llama_cpp import LlamaGrammar
GBNF_GRAMMAR = r'''
root ::= "HOLD" | "EXIT: " string
string ::= [^\n]+
'''
print("Creating grammar...")
try:
    grammar = LlamaGrammar.from_string(GBNF_GRAMMAR)
    print("Success")
except Exception as e:
    print(f"Failed: {e}")
