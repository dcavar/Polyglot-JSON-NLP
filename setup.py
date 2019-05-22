
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="polyglotjsonnlp",
    version="0.2.4",
    author="Damir Cavar, Oren Baldinger, Maanvitha Gongalla, Anurag Kumar, Murali Kammili, Boli Fang",
    author_email="damir@cavar.me",
    description="The Python Polyglot JSON-NLP package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dcavar/Polyglot-JSON-NLP",
    packages=setuptools.find_packages(),
    install_requires=[
        'polyglot>=16.7.4',
        'pyjsonnlp>=0.2.5'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    test_suite="tests",
    tests_require=["pytest", "coverage"]
)
