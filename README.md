# process_motion_tf

Command from raspberry pi (assume 192.168.1.27 is the IP address)

```
/usr/bin/raspivid -w 1920 -h 1080 -t 0 -fps 60 -x udp://192.168.1.27:9999 -n -o /dev/null -rf rgb
```

Command to run:

```
nc -ulp 9999 | python3 process_motion_tf.py 
```
