from slowapi import Limiter
from slowapi.util import get_remote_address

# Single shared limiter instance — imported by both routes.py and main.py.
limiter = Limiter(key_func=get_remote_address)
