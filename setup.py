from setuptools import setup
from setuptools.extension import Extension

try:
    from Cython.Build import cythonize
except ImportError:
    cythonize = None
try:
    import numpy
except ImportError:
    numpy = None

import versioneer


class NumpyIncludePath(object):
    """Lazy import of numpy to get include path."""
    @staticmethod
    def __str__():
        import numpy
        return numpy.get_include()


if cythonize is not None and numpy is not None:
    EXTENSIONS = cythonize([Extension('lib.functional',
                                      ['lib/functional.pyx'],
                                      include_dirs=[numpy.get_include()]),
                           ],
                           annotate=True,
                           compiler_directives={'language_level': 3})
else:
    EXTENSIONS = [Extension('lib.functional',
                            ['lib/functional.pyx'],
                            include_dirs=[NumpyIncludePath()])]


setup(
    name='lib',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=[
        'lib',
        'lib.datasets',
        'lib.decoder',
        'lib.decoder.generator',
        'lib.encoder',
        'lib.network',
        'lib.show',
        'lib.transforms',
        'lib.visualizer',
    ],
    license='GNU AGPLv3',
    description='Differentiable Multi-Granularity Human Representation Learning for Instance-Aware Human Semantic Parsing',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    author='Tianfei Zhou',
    author_email='ztfei.debug@gmail.com',
    url='https://github.com/tfzhou/MG-HumanParsing',
    ext_modules=EXTENSIONS,
    zip_safe=False,

    python_requires='>=3.6',
    install_requires=[
        'numpy>=1.16',
        'pysparkling',  # for log analysis
        'python-json-logger',
        'scipy',
        'torch>=1.3.1',
        'torchvision>=0.4',
        'pillow',
        'dataclasses; python_version<"3.7"',
    ],
    extras_require={
        'dev': [
            'flameprof',
            'jupyter-book>=0.7.0b',
            'matplotlib',
            'nbdime',
            'nbstripout',
        ],
        'onnx': [
            'onnx',
            'onnx-simplifier>=0.2.9',
        ],
        'test': [
            'nbval',
            'onnx',
            'onnx-simplifier>=0.2.9',
            'pylint',
            'pytest',
            'opencv-python',
            'thop',
        ],
        'train': [
            'matplotlib',  # required by pycocotools
            'pycocotools',  # pre-install cython (currently incompatible with numpy 1.18 or above)
        ],
    },
)
