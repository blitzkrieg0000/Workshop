import os
import inspect

def CWD():
    """Bu fonksiyonu çağıran dosyanın konumunu al"""
    p = "/"
    try: p = os.path.abspath(os.path.dirname(inspect.stack()[1][1]))
    except Exception as e: pass
    return p