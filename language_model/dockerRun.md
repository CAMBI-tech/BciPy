The following commands are to be executed in windows terminal

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

Now docker is up and running

-----------------------
2.
A ONE TIME BUILD of the IMAGE


```
docker build -t lmimage
```

Now the image is built

-----------------------
3.
When finishing working with docker
Stop the VM


```
docker-machine stop mymachine
```

Now docker is shut down
