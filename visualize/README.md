# Usage guide for visualization code

Currently only test1 and test2 work. I will fix test3 later.

Usage test1 : `python main.py test1 -m \<PATH TO MODEL PTH FILE\> -t \<LAYER\> -i \<DATA DIRECTORY\> -k \<NUMBER OF CLASSES TO VISUALIZE\>`
Usage test2 : `python main.py test2 -m \<PATH TO MODEL PTH FILE\> -i \<DATA DIRECTORY\> -t \<TARGET CLASS TO VISUALIZE\>`

Supported layers: relu, layer1, layer2, layer3, layer4
Data directory: /path/to/cat/
Classes to visualize: 35 for all classes
