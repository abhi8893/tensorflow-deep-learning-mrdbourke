from functools import wraps
 
class Default(object):
    def __init__(self, name):
        super(Default, self).__init__()
 
        self.name = name
 
 
def set_defaults(defaults):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            # Backup original function defaults.
            original_defaults = f.func_defaults
 
            # Replace every `Default("...")` argument with its current value.
            function_defaults = []
            for default_value in f.func_defaults:
                if isinstance(default_value, Default):
                    function_defaults.append(defaults[default_value.name])
                else:
                    function_defaults.append(default_value)
 
            # Set the new function defaults.
            f.func_defaults = tuple(function_defaults)
 
            return_value = f(*args, **kwargs)
 
            # Restore original defaults (required to keep this trick working.)
            f.func_defaults = original_defaults
 
            return return_value
 
        return wrapper
 
    return decorator

