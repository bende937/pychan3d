from distutils.core import setup

setup(name='pychan3d',
      version='0.1.0',
      description='scripting library for channel network modeling of flow and transport in fractured media',
      url='https://github.com/bende937/pychan3d',
      author='Benoit Dessirier',
      author_email='benoit.dessirier@geo.uu.se',
      license='MIT',

      packages=['pychan3d', 'examples/example1', 'examples/example2', 'examples/example3'],
      requires=['numpy', 'scipy', 'pyamg'],
      package_data={'pychan3d': ['../LICENSE.txt'],
                    'examples/example1': ['*'],
                    'examples/example2': ['*'],
                    'examples/example3': ['*']})
