from distutils.core import setup

setup(
    name='utix',
    packages=['utix'],
    version='0.1',  # Start with a small number and increase it with every change you make
    license='MIT',  # Chose a license from here: https://help.github.com/articles/licensing-a-repository
    description='initial release',  # Give a short description about your library
    author='Xinli Yu',  # Type in your name
    author_email='tuf72841@temple.edu',  # Type in your E-Mail
    url='https://github.com/XinliYu/utix',  # Provide either the link to your github or to your website
    download_url='https://github.com/user/reponame/archive/v_01.tar.gz',  # I explain this later on
    keywords=['utility functions', 'tools', 'science', 'experiments'],  # Keywords that define your package best
    install_requires=[
        'numpy',
        'nltk'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',  # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        'Intended Audience :: Scientists, Developers',  # Define that your audience are developers
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)
