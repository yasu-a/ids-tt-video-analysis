class Locator:
    def __init__(self, lst):
        self.__lst = lst

    def __getitem__(self, locator: slice):
        i = locator.start or 0
        while locator.stop is None or i < locator.stop:
            try:
                item = self.__lst[i]
            except IndexError:
                break
            yield i, item
            i += locator.step or 1

