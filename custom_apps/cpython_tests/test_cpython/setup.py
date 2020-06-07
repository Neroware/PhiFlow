from distutils.core import setup, Extension

module = Extension("testlib", sources = ["testlib.c"])

setup(name="testpck", 
        version="1.0", 
        description="This is a test package for test modules",
        ext_modules = [module])
