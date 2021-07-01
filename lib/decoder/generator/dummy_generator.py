from .generator import Generator


class DummyGenerator(Generator):
    def __init__(self):
        pass

    def __call__(self, fields):
        pass