# Cybertron-Data

This repository contains the data set for **Cybertron**.

## How it works
We collect JavaScript files (about 11005 files) from several famous GitHub repositories like AngularJS, Node and ThreeJs.
Then we use some JavaScript obfuscators to obfuscate the original files.
We also cache the tokens and AST for each file so that they can be immediately used by any machine learning model.
We also provide some JavaScript functions for processing and analyzing the data set.

In order to use the JavaScript functions, please run `yarn install`

## Obfuscators

- [JavaScript-Obfuscator](https://github.com/javascript-obfuscator/javascript-obfuscator)


## Data "structure"

| File/Folder  | Purpose |
| ------------ | ------------- |
| origin       | This folder contains JavaScript files fetched from other GitHub repositories. |
| index.js     |  A JavaScript that contains methods to generate base lines between original file and obfuscated file. |
| utils.js     |  A JavaScript that contains utility methods for data processing and analyzing. |

