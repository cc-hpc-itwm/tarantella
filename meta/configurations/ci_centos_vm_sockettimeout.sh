# https://vincent.bernat.ch/en/blog/2014-tcp-time-wait-state-linux
# https://codetips.coloza.com/2015/11/reduce-timewait-socket-connections.html

# Reset timeout to 2 seconds (default: 60)
# Comments contain default values
# Must be executed as root
# sysctl -w net.ipv4.tcp_fin_timeout=30
# net.ipv4.tcp_fin_timeout = 60
echo 2 > /proc/sys/net/ipv4/tcp_fin_timeout

# Enable reuse/recycle (default: 0)
# net.ipv4.tcp_tw_reuse = 0
echo 1 > /proc/sys/net/ipv4/tcp_tw_reuse 

# net.ipv4.tcp_tw_recycle = 0
#echo 1 > /proc/sys/net/ipv4/tcp_tw_recycle 


# net.ipv4.tcp_keepalive_intvl = 75
echo 30 > /proc/sys/net/ipv4/tcp_keepalive_intvl

# net.ipv4.tcp_keepalive_probes = 9
echo 5 > /proc/sys/net/ipv4/tcp_keepalive_probes


