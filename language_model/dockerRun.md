1.
```{r, engine='bash', count_lines}
docker-machine start mymachine
```
Get the environment commands for your  VM

```{r, engine='bash', count_lines}
docker-machine env --shell cmd mymachine > env.txt
```
read from env.txt to export
```{r, engine='bash', count_lines}
FOR /f "tokens=*" %%i IN (env.txt) DO %%i
DEL env.txt
```
NOW DOCKER CAN RUN
-----------------------
2.
A ONE TIME BUILD of the IMAGE
```{r, engine='bash', count_lines}
docker build -t lmimage
```
-----------------------
3.
WHEN FINISHING WORKING WITH DOCKER
Stop the VM
```{r, engine='bash', count_lines}
docker-machine stop mymachine
```
