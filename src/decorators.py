from pydash import flow, to_list
import pydash as _

def apply(map_result=None, on_args=None, on_result=None):
    def decorator(f):
        def wrapper(*args, **kwarg):
            flow(*to_list(on_args or []))(*args, **kwarg)
            result = flow(*to_list(map_result or _.identity))(f(*args, **kwarg))
            flow(*to_list(on_result or []))(result)
            return result
        return wrapper
    return decorator
