# ATLAS-Interactive — Setup for Beginners
!! If you do not have a .md viewer, copy paste this text into : https://markdownlivepreview.com.

No programming knowledge needed. Follow the steps in order.

## 1\. Install Miniconda

Download and run the installer (this also gives you Python — you don't need to install Python separately):

* **Windows:** https://www.anaconda.com/download/success → *Miniconda Installers* → Windows 64-bit
* **Mac:** same page → Miniconda for macOS
* Click *Next* through the installer, keep all default options.

## 2\. Open the command prompt

* **Windows:** press the Start button, type `Anaconda Prompt`, open it.
* **Mac:** open the `Terminal` app.

You now see a black/white window where you type commands. Type each command below and press **Enter** after it.

## 3\. Create an environment

An "environment" is a private box that holds the program's building blocks.

```
conda create -n atlas python=3.10
```

Type `y` and Enter when it asks. Then turn it on:

```
conda activate atlas
```

The line now starts with `(atlas)`. It must say this every time you use the program.

## 4\. Open the Command Prompt in Windows

`cd` means "change directory" — it moves you into a folder.

Find the `ATLAS-mod` folder in your file explorer, click the address bar at the top, and copy the path. Then type `cd`, a space, and paste the path in quotes:

```
cd "C:\...\Atlas-mod"
```

On Windows, if the folder is on a different drive than `C:`, first type the drive letter alone:

```
D:
```

Type `dir` (Windows) or `ls` (Mac) and press Enter — you should see `gui`, `workspace`, `README.md`.

## 5\. Install the required packages

Only needed once. Both lines, one at a time:

```
pip install torch torchvision
pip install -e .
```

This takes a few minutes.

## 6\. Start the program

Every time from now on: open the command prompt, then

```
conda activate atlas
```

```
cd ":C\...\Atlas_mod"
```

```
python gui.py
```

The first run downloads the model files automatically — this takes a while. After that the window opens and the rest is self-explanatory. See [TIPS](gui/TIPS.md), also shown inside the tool.

## If something goes wrong

* `conda is not recognized` → you're in the wrong window, use **Anaconda Prompt**.
* `No such file or directory` → the path after `cd` is wrong, copy it again from the file explorer.
* Line doesn't start with `(atlas)` → run `conda activate atlas`.

