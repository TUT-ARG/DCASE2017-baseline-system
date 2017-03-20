#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Decorators
==================
Classes for file handling


"""

import functools


def before_and_after_function_wrapper(func):
    """Before and after decorator

    Check if class method has before and after methods and call them.
    Idea by Konstantinos Drossos <konstantinos.drossos@tut.fi>

    Parameters
    ----------
    func

    Returns
    -------

    """

    @functools.wraps(func)
    def function_wrapper(*args, **kwargs):
        self_obj = args[0]
        before_func = getattr(self_obj, '_before_{}'.format(func.__name__), None)
        after_func = getattr(self_obj, '_after_{}'.format(func.__name__), None)

        if before_func is not None:
            before_func(*args, **kwargs)

        to_return = func(*args, **kwargs)

        if after_func is not None:
            after_func(to_return)

        return to_return

    return function_wrapper

