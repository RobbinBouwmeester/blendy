import datetime

"""


Credits for this script go to:

    https://gist.github.com/igniteflow/1253276


"""

class Timer(object):
    """A simple timer class"""
    
    def __init__(self):
        pass
    
    def start(self,message="Start: "):
        """Starts the timer"""
        self.start = datetime.datetime.now()
        return "<%s> %s" % (self.start,message)
    
    def stop(self, message="Total: "):
        """Stops the timer.  Returns the time elapsed"""
        self.stop = datetime.datetime.now()
        return message + str(self.stop - self.start)
    
    def now(self, message="Now: "):
        """Returns the current time with a message"""
        return message + ": " + str(datetime.datetime.now())
    
    def elapsed(self, message="Elapsed: "):
        """Time elapsed since start was called"""
        return "<%s|%s> %s" % (datetime.datetime.now(),(datetime.datetime.now() - self.start),message)
    
    def split(self, message="Split started at: "):
        """Start a split timer"""
        self.split_start = datetime.datetime.now()
        return message + str(self.split_start)
    
    def unsplit(self, message="Unsplit: "):
        """Stops a split. Returns the time elapsed since split was called"""
        return message + str(datetime.datetime.now() - self.split_start)