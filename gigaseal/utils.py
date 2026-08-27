## This script is for general programmatic tools. Patch-specific tools should go in patch_utils.py, and analysis-specific tools should go in the analysis module.
import numpy as np
import pandas as pd
import sys
import functools
import inspect
import numpy as np

DEBUG = True


def debug_wrap(func=None, returns=None):
    """
    Decorator that wraps functions in try-except when DEBUG=False.
    When DEBUG=True, exceptions are raised normally for easier debugging.
    On failure it returns np.nan values shaped to match the function's return.

    Usage:
        @debug_wrap                     # infer shape from the return type hint
        @debug_wrap(returns=3)          # force a 3-tuple of np.nan
        @debug_wrap(returns=(0, 0))     # force a 2-tuple of np.nan
    """
    # Support both @debug_wrap and @debug_wrap(returns=...) forms.
    if func is None:
        return functools.partial(debug_wrap, returns=returns)

    def make_nan_from_typehint(typehint):
        if hasattr(typehint, '__origin__') and typehint.__origin__ is tuple:
            args = typehint.__args__
            if Ellipsis in args:
                return np.nan
            return tuple(make_nan_from_typehint(arg) for arg in args)
        return np.nan

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if DEBUG:
            # Debug mode: let exceptions propagate for full traceback
            return func(*args, **kwargs)
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"Warning: {func.__name__} failed with {type(e).__name__}: {e}")

            # Explicit override wins over the type hint.
            if returns is not None:
                if isinstance(returns, int):
                    return tuple(np.nan for _ in range(returns))
                if isinstance(returns, (tuple, list)):
                    return tuple(np.nan for _ in returns)

            return_annotation = inspect.signature(func).return_annotation
            if getattr(return_annotation, '__origin__', None) is tuple:
                return make_nan_from_typehint(return_annotation)

            # if all else fails here is a single nan
            return np.nan
    return wrapper


class arg_wrap(object):
    """
    Wraps the argparser to catch any exceptions and query the user for the input again
    """

    def __init__(self, argparser, cli_prompt=True, gui_prompt=False, **kwargs):
        # determine which args are needed
        self.argparser = argparser
        self.required_args = [
            action.dest for action in self.argparser._actions if action.required]
        self.optional_args = [
            action.dest for action in self.argparser._actions if not action.required]
        self.parse_args = None

        if cli_prompt and gui_prompt:
            raise ValueError("Both cli_prompt and gui_prompt cannot be True")
        elif cli_prompt:
            self.parse_args = self._prompt_cli
        elif gui_prompt:
            self.parse_args = self._prompt_gui

    def __call__(self):
        return self.parse_args()

    def _determine_missing_args(self):
        missing_args = []
        sys_input = sys.argv
        for arg in self.required_args:
            if arg not in sys_input:
                missing_args.append(arg)
        return missing_args

    def _prompt_cli(self):
        try:  # try to parse the args
            args = self.argparser.parse_args()
        except:
            missing_args = self._determine_missing_args()
            args = self._query_args()

        return args

    def _query_args(self):
        args = {}
        for arg in self.required_args:
            args[arg] = input(f"Enter the value for {arg}: ")
        for arg in self.optional_args:
            args[arg] = input(f"Enter the value for {arg} (optional): ")
        return args

    def _prompt_gui(self):
        raise NotImplementedError("GUI not yet implemented")
