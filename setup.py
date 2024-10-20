from setuptools import find_packages,setup

setup(
    name='mcq_generator',
    version='0.0.1',
    author='Raso',
    author_email='raso@gmail.com',
    install_requires=["openai", "langchain", "streamlit", "python-dotenv", "PyPDF2"],
    package=find_packages()
)