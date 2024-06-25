How to install

clone the repo 

download pyenv
curl https://pyenv.run | bash
or 
brew install pyenv

First, let's add pyenv to your shell configuration. Since you're using SSH to access a Linux machine, we'll assume you're using bash. We'll modify both ~/.bashrc (for interactive non-login shells) and ~/.profile (for login shells) to ensure pyenv works in all scenarios.
Run these commands:

echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc

echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.profile
echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.profile
echo 'eval "$(pyenv init -)"' >> ~/.profile

If you want to use pyenv-virtualenv, also add this line to your ~/.bashrc:
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc

After making these changes, you need to reload your shell configuration. You can do this by running:
source ~/.bashrc

download poetry 
pip install poetry 

After installation, you need to add Poetry to your PATH. Add this line to your ~/.bashrc and ~/.profile:
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.profile

Reload your shell configuration again:
source ~/.bashrc

Verify that both pyenv and poetry are installed correctly:
pyenv --version
poetry --version

Now install the python required for the job 
pyenv install 3.12.3

set it as the local python 
pyenv 3.12.3 local 
