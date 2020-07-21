# Sequential tracing for chromosome-wide imaging

## Installtion guide

by Pu Zheng

2020.06.15

Example code in this folder has been verified and tested by the following installation steps on Windows Server 2019 (should be similar in Windows 10)

1. Install [Anaconda3](https://repo.anaconda.com/archive/) ( version for 64-bit python 3.7 is recommended) and go through typical installation procedure. 

    This is what I got from conda info after installtion: 
    ```
        active environment : None 

          user config file : C:\Users\puzheng\.condarc 

    populated config files : C:\Users\puzheng\Anaconda3\condarc 

              conda version : 4.8.3 

        conda-build version : 3.18.11 

            python version : 3.7.6.final.0 

          virtual packages : 

          base environment : C:\Users\puzheng\.local\conda_root  (read only) 

              channel URLs : https://your.repo/binstar_username/win-64 

                              https://your.repo/binstar_username/noarch 

                              http://some.custom/channel/win-64 

                              http://some.custom/channel/noarch 

                              https://repo.anaconda.com/pkgs/main/win-64 

                              https://repo.anaconda.com/pkgs/main/noarch 

                              https://repo.anaconda.com/pkgs/r/win-64 

                              https://repo.anaconda.com/pkgs/r/noarch 

                              https://repo.anaconda.com/pkgs/msys2/win-64 

                              https://repo.anaconda.com/pkgs/msys2/noarch 

              package cache : C:\Users\puzheng\my-pkgs 

                              C:\opt\anaconda\pkgs 

          envs directories : C:\Users\puzheng\my-envs 

                              C:\opt\anaconda\envs 

                              C:\Users\puzheng\.conda\envs 

                              C:\Users\puzheng\.local\conda_root\envs 

                              C:\Users\puzheng\AppData\Local\conda\conda\envs 

                  platform : win-64 

                user-agent : conda/4.8.3 requests/2.22.0 CPython/3.7.6 Windows/10 Windows/10.0.17763 

              administrator : False 

                netrc file : None 

              offline mode : False 
    ```

2. Clone the Chromatin_Analysis_2020_cell repository through: 

    ```
    git clone git@github.com:ZhuangLab/Chromatin_Analysis_2020_cell.git
    ```

3. Open the corresponding terminal (or cmd), install required packages in your preferred environment by: 

    ```
    conda install biopython 
    pip install opencv-python 
    ```

4. If you hope to reproduce [PostAnalysis](https://github.com/ZhuangLab/Chromatin_Analysis_2020_cell/tree/master/sequential_tracing/PostAnalysis) Download data from: Zenodo with **DOI:10.5281/zenodo.3928890**, open jupyter and run through all boxes. 


>Test Log
>
>>Tested by Pu Zheng in Windows 10 Enterprise, version 1809, OS build 17763.720, 2020.06.15.
>>
>>Updated by Pu Zheng, 2020.06.30, 2020.07.01, 2020.07.21,

 

 