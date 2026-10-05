import importlib.util
import os


def load_class_from_file(file_path, class_name):
    if not file_path or not os.path.isfile(file_path):
        raise FileNotFoundError(f"file not found: {file_path}")

    spec = importlib.util.spec_from_file_location(class_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not read the module file")

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        raise ImportError(str(error)) from error

    if not hasattr(module, class_name):
        raise AttributeError(f"class '{class_name}' is not defined in that file")
    return getattr(module, class_name)