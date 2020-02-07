============
setup macbook
============

Install Home Brew::

    $ ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
    $ brew doctor

Install Maven::

    $ brew update             # Fetch latest version of homebrew and formula.
    $ brew search maven       # Searches all known maven for a partial or exact match.
    $ brew info maven         # Displays information about the given maven.
    $ brew install maven      # Install the given maven.
    $ brew cleanup            # Remove any older versions from the cellar.

Install git::

    $ brew install git         # Install the given git.

Install Java::

    $ brew cask install java    # This will install the latest jdk:
    $ brew cask install java8   # This will install the jdk8:

Install Python::

    $ brew install python3      # This will install the latest python:
    $ python3 --version         # Check Python3 version
