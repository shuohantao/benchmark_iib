import os
import importlib

model_files = [f for f in os.listdir(os.path.dirname(__file__)) if f.endswith('.py') and f != '__init__.py']

for file in model_files:
    if file.startswith('_'):
        continue
    module_name = file[:-3]  # Remove .py extension
    module = importlib.import_module('.' + module_name, package='models')

    # Assuming the class name is the same as the file name
    class_obj = getattr(module, module_name)

    # Add the class to the current namespace
    globals()[module_name] = class_obj
