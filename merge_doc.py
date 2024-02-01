import os
import json
import re

json_data = json.load(open("jittor_misc.py.json", "r"))
origin_code = open("misc.py", "r").read()


# import inspect

for func in json_data:
    pattern = r'def\s+' + func["func_name"] + r'\s*\([^)]*\):\s*("""[^"]*"""|\'\'\'[^\']*\'\'\')'
    matches = re.findall(pattern, origin_code, re.DOTALL)
    if func["func_name"] == "reshape": 
        print(pattern, matches)
    new_docstring = func["docstring"]
    if len(matches) != 0:
        for mm in matches:
            origin_code = origin_code.replace(mm, "\'\'\'" + new_docstring + "\'\'\'")
    else:
        print(func["func_name"])
        # print(func["func_name"], len(matches), origin_code.replace(matches[0], "\"" + new_docstring + "\""))
        # exit()
    # pattern = "def\s+" + func["func_name"] + r"\(.*?\):\s*'''(.*?)'''"
    # print(pattern)
    # pattern = re.compile("def\s+" + func["func_name"] + r"\(.*?\):\s*'''(.*?)'''", re.S)
    # matches = re.findall(pattern, origin_code_path, re.DOTALL)
    # match = pattern.findall(origin_code_path)
    # funcs = globals()[func["func_name"]]
    # source = inspect.getsource(funcs)
    # old_docstring = inspect.getdoc(funcs)

    # # print(func["func_name"], old_docstring)
    # origin_code = origin_code.replace(f"'''{old_docstring}'''", f"\"{new_docstring}\"")
    # print(old_docstring, source.find(old_docstring))    
f = open("new_doc.py", "w")
f.write(origin_code)