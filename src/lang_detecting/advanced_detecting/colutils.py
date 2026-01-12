from collections import OrderedDict


def order_dict_to_dict(d):
    match d:
        case OrderedDict(): return {k: order_dict_to_dict(v) for k, v in d.items()}
        case list() | tuple() | set(): return [order_dict_to_dict(e) for e in d]
        case str() | int() | float(): return d
        case _: return dict(d)
