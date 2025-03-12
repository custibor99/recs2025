# Recommender Systems 2025 Project
Based on the [RecSys 2025 Challange](https://www.recsyschallenge.com/2025/#organizers)

## Technical details
Based on **python 3.13**
 
Dependency management with [uv](https://docs.astral.sh/uv/guides/install-python/) 

### How to use and install uv
On macOs and Linux
~~~
curl -LsSf https://astral.sh/uv/install.sh | sh

or

wget -qO- https://astral.sh/uv/install.sh | sh
~~~

On windows
~~~
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
~~~

Once installed, we can prepare python versions
~~~
# Make sure uv installed python versions exist
# Note installing python binaries like this is still a "preview" feature
uv python install --preview 3.13 3.12
 
# Set a default python version to use
uv python install --preview 3.13 --default
~~~

How to add new dependencies
~~~
uv add <dependency name>
~~~

How to run create a virtual enviroment
~~~
uv venv
~~~

How to run a script
~~~
uv run python hello.py
~~~
### Code formating
Before commiting your code please format it using ruff
~~~
uv run ruff format
~~~
## Download data and init repository
To download and split the data run the following code. Make sure python has permisions to access your directories
~~~
uv run python init_repo.py
~~~
