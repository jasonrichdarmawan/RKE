def is_notebook() -> bool:
    try:
        __IPYTHON__ # type: ignore
        return True
    except NameError:
        return False