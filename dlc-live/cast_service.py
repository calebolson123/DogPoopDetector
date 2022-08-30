import time
import pychromecast

class CastSoundService():

    def __init__(self):
        self.connected = False
        self.cast = None
        self.connect()


    def connect(self):
        while not self.connected:
            print('Attempting connection to speaker...')
            chromecasts, browser = pychromecast.get_listed_chromecasts(friendly_names=["Your Device"])

            if len(chromecasts) == 0:
                continue

            self.cast = chromecasts[0]
            self.cast.wait()
            if self.cast.status:
                self.connected = True
                print(self.cast.status)


    def debounce(s):
        """Decorator ensures function that can only be called once every `s` seconds.
        """
        def decorate(f):
            t = None

            def wrapped(*args, **kwargs):
                nonlocal t
                t_ = time.time()
                if t is None or t_ - t >= s:
                    result = f(*args, **kwargs)
                    t = time.time()
                    return result
            return wrapped
        return decorate


    @debounce(4)
    def play_sound(self):
        print('Play sound request triggered...')
        self.cast.media_controller.play_media('http://ip_of_your_device/file_on_device.mp4', 'video/mp4')
        self.cast.media_controller.block_until_active()
        self.cast.media_controller.play()