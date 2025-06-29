# AI

# Instructions

1. Install WSL2 UBUNTU
2. Install Visual Studio Code in Windows
3. Clone the Repository to your home directory in Linux
4. In Linux:
   1. Execute command: `bash 1.install_packages.sh`
   2. Execute command: `bash 2.create_virtual.environment.sh`
   3. Execute command: `bash 3.activate_virtual_environment.sh`
   4. Execute command: `source venv/bin/activate`

## Docker (skip if you already have docker installed)

   5. Install docker with `bash docker-install.sh` 
   6. Add your user to the docker group: `sudo usermod -aG docker $USER`
   7. logout and login on your linux again... `exit`

## Start docker service
   8. activate docker service: `service docker start`
   9. Execute command to open visual studio code in Linux: `code .`
   10. rename file `.env_template` to `.env`
5. Access [OpenAI](https://openai.com/) and create an API KEY and copy paste it to `.env`file
6. Access [GroqCloud](https://console.groq.com/keys),create an API KEY and copy and paste it to the `.env` file
7. Access [Anthropic](https://console.anthropic.com/),create an API KEY and copy and paste it to the `.env` file)
8. Access [TavilySearch](https://www.tavily.com/), create an API KEY and copy and paste it to the `.env` file

### Recommended Extensions


1. Name: Jupyter
   Id: ms-toolsai.jupyter
   Description: Jupyter notebook support, interactive programming and computing that supports Intellisense, debugging and more.
   Version: 2024.3.1
   Publisher: Microsoft
   VS Marketplace Link: https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter

2. Name: Python
   Id: ms-python.python
   Description: Python language support with extension access points for IntelliSense (Pylance), Debugging (Python Debugger), linting, formatting, refactoring, unit tests, and more.
   Version: 2024.4.1
   Publisher: Microsoft
   VS Marketplace Link: https://marketplace.visualstudio.com/items?itemName=ms-python.python

