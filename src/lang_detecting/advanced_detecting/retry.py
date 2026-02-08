from typing import Callable, Collection


def retry_on(func: Callable, exceptions: Collection[type[Exception]] | type[Exception] | None, n_tries: int = 5, *args, **kwargs):
    match exceptions:
        case None: exceptions = [Exception]
        case Exception(): exceptions = [exceptions]
        case _: pass

    for i in range(n_tries):
        try:
            func(*args, **kwargs)
            break
        except exceptions as e:
            pass
    else:
        raise RuntimeError(f'Met max retries {n_tries}') from e
