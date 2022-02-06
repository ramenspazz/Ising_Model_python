import traceback  # noqa F401


class Watcher(object):
    def __init__(self, obj=None, attr=None, log_file='log.txt', include=[],
    enabled=False):  # noqa E128
        """
            Debugger that watches for changes in object attributes

            Parameters
            ----------
            obj - object to be watched
            attr - string, name of attribute
            log_file - string, where to write output
            include - list of strings, debug files only in these directories.
                Set it to path of your project otherwise it will take long time
                to run on big libraries import and usage.
        """

        self.log_file = log_file
        with open(self.log_file, 'wb'):
            pass
        self.prev_st = None
        self.include = [incl.replace('\\', '/') for incl in include]
        if obj:
            self.value = getattr(obj, attr)
        self.obj = obj
        self.attr = attr
        self.enabled = enabled  # Important, must be last line on __init__.

    def __call__(self, *args, **kwargs):
        kwargs['enabled'] = True
        self.__init__(*args, **kwargs)

    def check_condition(self):
        tmp = getattr(self.obj, self.attr)
        result = tmp != self.value
        self.value = tmp
        return result
