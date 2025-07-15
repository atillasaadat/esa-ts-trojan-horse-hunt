# ESA Time Series Trojan Horse Hunt

## About
A competition submission for the European Space Agency (ESA) Time Series Trojan Horse Hunt challenge for the CS7643 Deep Learning course at Georgia Tech.

[Competition Link](https://www.kaggle.com/competitions/trojan-horse-hunt-in-space)

[Competition Paper](https://arxiv.org/abs/2506.01849)

[Competition Dataset Link](https://www.kaggle.com/competitions/trojan-horse-hunt-in-space/data)

## Setup

First install UV from [here](https://docs.astral.sh/uv/getting-started/installation/), I personally recommend installing it using [Homebrew](https://brew.sh/).

```bash
uv sync
```

## Usage

```bash
uv run main.py
```

## Setup Data

After UV setup, setup your Kaggle API to download dthe data:

To use the Kaggle API, sign up for a Kaggle account at https://www.kaggle.com. Then go to the 'Account' tab of your user profile (`https://www.kaggle.com/<username>/account`) and select 'Create API Token'. This will trigger the download of `kaggle.json`, a file containing your API credentials.
Place this file in the location appropriate for your operating system:
* Linux: `$XDG_CONFIG_HOME/kaggle/kaggle.json` (defaults to `~/.config/kaggle/kaggle.json`). The path `~/.kaggle/kaggle.json` which was used by older versions of the tool is also still supported.
* Windows: `C:\Users\<Windows-username>\.kaggle\kaggle.json` - you can check the exact location, sans drive, with `echo %HOMEPATH%`.
* Other: `~/.kaggle/kaggle.json`

Then, run the `setup_data.sh` to download the data into the `./data` folder