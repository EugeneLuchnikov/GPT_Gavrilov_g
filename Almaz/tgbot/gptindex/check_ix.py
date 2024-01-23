from utils.initconf_env import load_config
from gptindex.ixapp import ixapp
from gptindex.gptutils import *
load_config()

ixapp.init()
print(ixapp.promt, num_tokens_from_string(ixapp.promt))
