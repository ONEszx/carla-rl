# 生成汽车流
import glob
import os
import sys
# ==============================================================================
# -- Find CARLA module ---------------------------------------------------------
# ==============================================================================

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# ==============================================================================
# -- Add PythonAPI for release mode --------------------------------------------
# ==============================================================================
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')
    sys.path.append("../examples/")
except IndexError:
    pass

import carla

# Connect to the client and retrieve the world object
try:
    client = carla.Client('127.0.0.1', 2000)
    for i in range(len(client.get_available_maps())):
         print(client.get_available_maps()[i])
except IndexError:
    pass
