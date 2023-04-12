# 💾 Installation
StreamingDataset installs via pip. Installing StreamingDataset in a virtual environment is recommended to avoid any dependency conflicts. StreamingDataset has been tested on Python 3.8, 3.9, and 3.10.

## Create a virtual environment
1. Create and navigate to your project directory:
    ```
    mkdir custom-project
    cd custom-project
    ```

2. Create a virtual environment inside your project directory:
    ```
    python -m venv <name_of_virtualenv>

    # For example
    python -m venv venv_streaming
    ```

3. Activate the virtual environment:
    ```
    source venv_streaming/bin/activate
    ```

## Install using pip
StreamingDataset can be installed using pip as follows:
```
pip install mosaicml-streaming
```

Run the following command to ensure the proper installation of the StreamingDataset. The following command prints the StreamingDataset version.
```
python -c "import streaming; print(streaming.__version__)"
```

## Install from source
Building and installing StreamingDataset from the source allows you to change the code base.
```
git clone https://github.com/mosaicml/streaming.git
cd streaming
pip install -e .
```
Run the following command to ensure the proper installation of the StreamingDataset. The following command prints the StreamingDataset version.
```
python -c "import streaming; print(streaming.__version__)"
```

That's it! Check out our [Quick Start Guide](quick_start.md) on using the StreamingDataset.
