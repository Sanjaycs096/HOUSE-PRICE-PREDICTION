from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_talisman import Talisman

# App-level shared extensions
limiter = Limiter(key_func=get_remote_address)
talisman = Talisman()
