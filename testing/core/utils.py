from pydash import flow


def apply(*map_funcs):
    def decorator(f):
        def wrapper(*args, **kwarg):
            return flow(*map_funcs)(f(*args, **kwarg))
        return wrapper
    return decorator


