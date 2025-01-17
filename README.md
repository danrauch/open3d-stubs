# open3d-stubs

Type stubs for Open3D. Work in progress - if types are missing add an Issue.
Baseline created via [pybind11-stubgen](https://github.com/sizmailov/pybind11-stubgen).

## Install (pylance)

### Specific VS Code project

Put open3d-stubs in a folder called `./typings` in your VS code project workspace.

### VS Code global

Put open3d-stubs in the pylance extension stubs folder (Windows default: %USERPROFILE%\.vscode\extensions\ms-python.vscode-pylance-xxxx.xx.xx\dist\bundled\stubs). You might have to repeat this process for every new pylance version.

### For VIM/Neovim

For Vim/Neovim users, check [coc-pyright](https://github.com/fannheyward/coc-pyright) for more information.

### Alternative

If none of the other approaches are working, place the stubs next to the installed open3d files in your current virtual environment or your Python site-package folder.