The following commands are to be executed in windows terminal
-----------------------

1.
Start a machine to turn on Docker
in theis case the machine is called mymachine
but you can naem it whatever you perfer


```
docker-machine start mymachine
```


Get the environment commands for your  VM


```
docker-machine env --shell cmd mymachine > env.txt
```

read from env.txt to export

```
FOR /f "tokens=*" %%i IN (env.txt) DO %%i
DEL env.txt
```

NOW DOCKER CAN RUN
-----------------------
2.
A ONE TIME BUILD of the IMAGE


```
docker build -t lmimage
```

-----------------------
3.
WHEN FINISHING WORKING WITH DOCKER
Stop the VM


```
docker-machine stop mymachine
```
