import os
import _thread
from live import PoopDetector
from cast_service import CastSoundService

# run basic http server on this device for serving up/accessing 
# marked up screenshot containing dog pooping
def start_server():
    os.system("python -m http.server")

_thread.start_new_thread(start_server, ())

cast_service = None
# cast_service = CastSoundService() # optionally cast audio to device on poop detection

poop_detector = PoopDetector(cast_service)
poop_detector.beefy_boy()